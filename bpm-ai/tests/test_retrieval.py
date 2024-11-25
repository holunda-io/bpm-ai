import pytest
from bpm_ai_core.llm.common.message import AssistantMessage
from bpm_ai_core.testing.fake_llm import FakeLLM
from bpm_ai_core.retrieval.retrieval import DocumentRetrieval, RetrievalResult, DocumentMatch
from bpm_ai_core.web_crawling.web_crawler import WebCrawler, CrawlingResult
from bpm_ai_inference.retrieval import ByaldiDocumentRetrieval
from bpm_ai_inference.web_crawling.playwright_crawler import PlaywrightWebCrawler

from bpm_ai.common.errors import MissingParameterError
from bpm_ai.retrieval.retrieval import retrieve_llm


class FakeRetrieval(DocumentRetrieval):
    def __init__(self, has_index=False):
        self._has_index = has_index
        self.indexed_files = []

    async def _do_index(self, file_path: str, index_name: str, metadata=None) -> None:
        self.indexed_files.append((file_path, index_name))

    async def _do_query(self, query: str, index_name: str, top_k: int = 3) -> RetrievalResult:
        return RetrievalResult(
            matches=[
                DocumentMatch(
                    file_path="test_image.jpg",
                    score=0.8
                )
            ]
        )

    async def has_index(self, index_name: str) -> bool:
        return self._has_index


class FakeCrawler(WebCrawler):
    async def _do_crawl(self, urls, depth=1, screenshot_dir=None) -> CrawlingResult:
        return CrawlingResult(
            screenshot_paths=["screenshot.png"],
            visited_urls=urls
        )


async def test_retrieve(llm):
    llm = llm or FakeLLM(
        name="openai",
        responses=[
            AssistantMessage(
                content="The image shows a test image."
            )
        ]
    )
    
    result = await retrieve_llm(
        llm=llm,
        input_data={
            #"strom": ["/Users/bennet/Documents/Dokumentenscans/Rechnungen/Vattenfall-strom-2022-23.pdf"],
            "elmshorn": ["https://de.wikipedia.org/wiki/Elmshorn"],
            "gettorf": ["https://de.wikipedia.org/wiki/Gettorf"],
            "eckernfoerde": ["https://de.wikipedia.org/wiki/Eckernförde"],
        },
        query="Welcher Bürgermeister hat seine Wahl deutlicher gewonenn, der von Elmshorn oder Eckernförde?",
        retrieval=ByaldiDocumentRetrieval(),
        crawler=PlaywrightWebCrawler()
    )

    if isinstance(llm, FakeLLM):
        # Verify LLM received the query
        llm.assert_last_request_contains("What's in the image?")

    #assert "image" in result["answer"]


async def test_retrieve_empty_input(llm):
    llm = llm or FakeLLM(name="openai")
    
    with pytest.raises(MissingParameterError):
        await retrieve_llm(
            llm=llm,
            input_data={},
            query="test query",
            retrieval=FakeRetrieval(),
            crawler=FakeCrawler()
        )

    # LLM should not be used if input is empty
    if isinstance(llm, FakeLLM):
        llm.assert_no_request()


async def test_retrieve_existing_index(llm):
    llm = llm or FakeLLM(
        name="openai",
        responses=[
            AssistantMessage(
                content="The image shows a test image."
            )
        ]
    )
    
    retrieval = FakeRetrieval(has_index=True)
    
    await retrieve_llm(
        llm=llm,
        input_data={
            "test_index": ["test_image.jpg"]
        },
        query="What's in the image??",
        retrieval=retrieval,
        crawler=FakeCrawler()
    )

    # Verify no indexing was done since index exists
    assert len(retrieval.indexed_files) == 0


async def test_retrieve_empty_query(llm):
    llm = llm or FakeLLM(name="openai")
    
    with pytest.raises(MissingParameterError):
        await retrieve_llm(
            llm=llm,
            input_data={"test_index": ["test_image.jpg"]},
            query="",
            retrieval=FakeRetrieval(),
            crawler=FakeCrawler()
        )

    # LLM should not be used if query is empty
    if isinstance(llm, FakeLLM):
        llm.assert_no_request()
