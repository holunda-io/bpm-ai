from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel

from bpm_ai_core.tracing.decorators import span
from bpm_ai_core.util.caching import cached, cachable


class CrawlingResult(BaseModel):
    """
    Result of web crawling containing screenshot file paths and metadata
    """
    screenshot_paths: List[str]
    visited_urls: List[str]


class WebCrawler(ABC):
    """
    Web Crawler that takes a list of URLs and crawling depth to capture screenshots
    """

    @abstractmethod
    async def _do_crawl(
            self,
            urls: List[str],
            depth: int = 1,
            screenshot_dir: str = None,
    ) -> CrawlingResult:
        """
        Implement the actual crawling logic in concrete classes

        Args:
            urls: List of starting URLs to crawl
            depth: How many levels deep to crawl (default: 1)
            screenshot_dir: Directory to save screenshots (default: None, will use system temp)

        Returns:
            CrawlingResult containing paths to screenshots and visited URLs
        """
        pass

    #@cached(exclude=["screenshot_dir"])
    @span(name="web-crawler")
    async def crawl(
            self,
            urls: List[str],
            depth: int = 1,
            screenshot_dir: str = None,
    ) -> CrawlingResult:
        """
        Crawl the given URLs up to specified depth and take screenshots

        Args:
            urls: List of starting URLs to crawl
            depth: How many levels deep to crawl (default: 1)
            screenshot_dir: Directory to save screenshots (default: None, will use system temp)

        Returns:
            CrawlingResult containing paths to screenshots and visited URLs
        """
        return await self._do_crawl(
            urls=urls,
            depth=depth,
            screenshot_dir=screenshot_dir
        )
