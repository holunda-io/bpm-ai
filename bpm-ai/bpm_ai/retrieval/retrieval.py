import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.llm.common.llm import LLM
from bpm_ai_core.retrieval.retrieval import DocumentRetrieval
from bpm_ai_core.util.image import blob_as_images
from bpm_ai_core.web_crawling.web_crawler import WebCrawler
from bpm_ai_core.tracing.decorators import trace
from bpm_ai_core.prompt.prompt import Prompt
from bpm_ai_core.util.file import is_supported_img_file

from bpm_ai.common.errors import MissingParameterError
from bpm_ai.common.multimodal import prepare_images_for_llm_prompt
from bpm_ai.retrieval.util import add_header_to_image

logger = logging.getLogger(__name__)


async def _synthesize_answer(llm: LLM, query: str, retrieved_docs: dict) -> str:
    """
    Synthesize an answer from retrieved documents using the LLM.
    
    Args:
        llm: LLM instance to use for answer synthesis
        query: Query to answer
        retrieved_docs: Dict mapping document identifiers to their content
        
    Returns:
        Synthesized answer string
    """
    prompt = Prompt.from_file(
        "answer_synthesis",
        context=retrieved_docs.values(),
        query=query
    )
    
    message = await llm.generate_message(prompt)
    return message.content


async def _determine_query_strategy(llm: LLM, query: str, input_data: dict, available_indexes: List[str]) -> List[dict]:
    """
    Determine which indexes to query and how to formulate the queries.
    
    Args:
        llm: LLM instance to use for strategy determination
        query: Original user query
        input_data: Optional user provided context information
        available_indexes: List of available index names

    Returns:
        List of dicts with index and query to execute
    """
    prompt = Prompt.from_file(
        "query_strategy",
        indexes=available_indexes,
        query=query,
        input_data=input_data
    )
    
    message = await llm.generate_message(prompt, output_schema={
        "type": "object",
        "properties": {
            "index_queries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "string", "description": "exact identifier of the index to query"},
                        "query": {"type": "string", "description": "query as fully-formed question to execute against the index"}
                    },
                    "required": ["index", "query"]
                }
            },
        }
    })
    
    return message.content["index_queries"]


async def _synthesize_final_answer(llm: LLM, query: str, input_data: dict, intermediate_answers: List[dict], output_schema: Optional[dict] = None) -> str | dict:
    """
    Synthesize a final answer from multiple intermediate answers.
    
    Args:
        llm: LLM instance to use for final synthesis
        query: Original user query
        input_data: Optional user provided context information
        intermediate_answers: List of dicts with index and content keys
        output_schema: Optional schema for structured JSON output

    Returns:
        Final synthesized answer as string or dict if json_output is True
    """
    prompt = Prompt.from_file(
        "final_synthesis",
        query=query,
        input_data=input_data,
        answers=intermediate_answers,
        json_output=output_schema is not None
    )
    
    if output_schema:
        message = await llm.generate_message(prompt, output_schema=output_schema)
        return message.content
    else:
        message = await llm.generate_message(prompt)
        return message.content


@trace("bpm-ai-retrieval", ["llm"])
async def retrieve_llm(
    llm: LLM,
    index: dict[str, List[str]],
    query: str,
    retrieval: DocumentRetrieval,
    crawler: WebCrawler,
    input_data: Optional[dict] = None,
    output_schema: Optional[dict] = None
) -> dict:
    """
    Retrieve relevant documents and synthesize an answer using an LLM.
    
    Args:
        llm: LLM instance to use for answer synthesis
        index: Dict mapping index names to lists of URLs/file paths
        query: User's question to answer
        retrieval: DocumentRetrieval instance for indexing/searching
        crawler: WebCrawler instance for processing URLs
        input_data: Optional user provided context information
        output_schema: Optional schema for structured JSON output

    Returns:
        Dict containing the synthesized answer
    """
    if not index:
        raise MissingParameterError("index is required")
    if not query:
        raise MissingParameterError("query is required")

    # Process each index
    for index_name, urls_or_paths in index.items():
        # Skip if index already exists
        if not await retrieval.has_index(index_name):
            await _create_index(index_name, retrieval, crawler, urls_or_paths)

    queries = await _determine_query_strategy(llm, query, input_data, list(index.keys()))
    
    # Execute each query individually and collect answers
    intermediate_answers = []
    for query_info in queries:
        # Retrieve documents for this query
        results = await retrieval.query(
            query=query_info["query"],
            index_name=query_info["index"],
            top_k=2
        )

        logger.info(results)
        
        # Prepare retrieved documents for LLM
        retrieved_docs = {
            f"doc_{i+1}": doc.file_path
            for i, doc in enumerate(results.matches)
        }
        retrieved_docs = prepare_images_for_llm_prompt(retrieved_docs)
        
        # Generate intermediate answer for this query
        answer = await _synthesize_answer(llm, query_info["query"], retrieved_docs)
        
        intermediate_answers.append({
            "index": query_info["index"],
            "content": answer
        })

    if len(intermediate_answers) == 0:
        return {"answer": None}

    # If only one answer, return it directly
    if len(intermediate_answers) == 1 and not output_schema:
        return {"answer": intermediate_answers[0]["content"]}
    
    # Otherwise synthesize final answer from intermediate answers
    final_answer = await _synthesize_final_answer(llm, query, input_data, intermediate_answers, output_schema)
    return {"answer": final_answer}


async def _create_index(index_name: str, retrieval: DocumentRetrieval, crawler: WebCrawler, urls_or_paths: List[str]):
    """
    Create an index from the given URLs or file paths.
    
    Args:
        index_name: Name of the index to create
        retrieval: DocumentRetrieval instance
        crawler: WebCrawler instance
        urls_or_paths: List of URLs or file paths to index
    """
    # Separate images and URLs
    images = [p for p in urls_or_paths if is_supported_img_file(p)]
    urls = [u for u in urls_or_paths if not is_supported_img_file(u)]
    image_paths = []
    
    # Process images and PDFs
    for image in images:
        blob = Blob.from_path_or_url(image)
        if blob.is_pdf():
            page_imgs = await blob_as_images(blob, accept_formats=["jpeg", "png"])
            temp_dir = tempfile.mkdtemp()
            pdf_name = Path(image).stem
            for i, img in enumerate(page_imgs):
                img = add_header_to_image(img, pdf_name, f"Page {i + 1}")
                filename = f"{pdf_name}_{i + 1}.jpeg"
                path = os.path.join(temp_dir, filename)
                img.save(path)
                image_paths.append(path)
        else:
            image_paths.append(image)
            
    # Crawl non-image URLs
    if urls:
        crawl_result = await crawler.crawl(urls=urls)
        # Add screenshot paths to images list
        image_paths.extend(crawl_result.screenshot_paths)
        
    # Index all images
    for img_path in image_paths:
        await retrieval.index(
            file_path=img_path,
            index_name=index_name
        )
