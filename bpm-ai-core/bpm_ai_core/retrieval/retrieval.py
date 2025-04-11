from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from bpm_ai_core.tracing.decorators import span

from pydantic import BaseModel


class DocumentMatch(BaseModel):
    """
    A single document match from a retrieval query
    """
    file_path: str
    score: float
    metadata: Optional[Dict] = None


class RetrievalResult(BaseModel):
    """
    Result of a retrieval query containing matched documents
    """
    matches: List[DocumentMatch]


class DocumentRetrieval(ABC):
    """
    Document Retrieval System for indexing and searching documents
    """

    @abstractmethod
    async def _do_index(
        self,
        file_path: str,
        index_name: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Index a single document with optional metadata
        
        Args:
            file_path: Path to the document to index
            index_name: Name of the index to store the document in
            metadata: Optional metadata associated with the document
        """
        pass

    @abstractmethod 
    async def _do_query(
        self,
        query: str,
        index_name: str,
        top_k: int = 3
    ) -> RetrievalResult:
        """
        Query an index for relevant documents
        
        Args:
            query: Query string to search for
            index_name: Name of the index to search in
            top_k: Number of top results to return
            
        Returns:
            RetrievalResult containing matched documents
        """
        pass

    @abstractmethod
    async def has_index(self, index_name: str) -> bool:
        pass

    async def index(
        self,
        file_path: str,
        index_name: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Index a single document with optional metadata
        
        Args:
            file_path: Path to the document to index
            index_name: Name of the index to store the document in
            metadata: Optional metadata associated with the document
        """
        await self._do_index(
            file_path=file_path,
            index_name=index_name,
            metadata=metadata
        )

    @span(name="retrieval")
    async def query(
        self,
        query: str,
        index_name: str,
        top_k: int = 3,
        score_threshold: Optional[float] = 0.0
    ) -> RetrievalResult:
        """
        Query an index for relevant documents
        
        Args:
            query: Query string to search for
            index_name: Name of the index to search in
            top_k: Number of top results to return
            score_threshold: Optional minimum score threshold for results
            
        Returns:
            RetrievalResult containing matched documents
        """
        result = await self._do_query(
            query=query,
            index_name=index_name,
            top_k=top_k
        )
        
        # Filter results by score threshold if provided
        if score_threshold is not None:
            result.matches = [m for m in result.matches if m.score >= score_threshold]
            
        return result
