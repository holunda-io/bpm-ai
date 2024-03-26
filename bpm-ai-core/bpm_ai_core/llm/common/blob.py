import contextlib
import mimetypes
import os
from io import BytesIO, BufferedReader
from pathlib import PurePath
from typing import Union, Optional, Dict, Any, Self, cast, Generator

import requests
from pydantic import BaseModel, Field, model_validator

from bpm_ai_core.util.storage import read_file_from_azure_blob, read_file_from_s3, is_s3_url, is_azure_blob_url


class Blob(BaseModel):
    """Blob represents raw data by either reference or value.

    Provides an interface to materialize the blob in different representations.

    Based on:
    https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/document_loaders/blob_loaders.py
    """

    data: Union[bytes, str, None]
    """Raw data associated with the blob."""

    mimetype: Optional[str] = None
    """MimeType not to be confused with a file extension."""

    path: Optional[Union[str, PurePath]] = None
    """Location where the original content was found."""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """Metadata about the blob (e.g., source)"""

    class Config:
        arbitrary_types_allowed = True
        frozen = True

    @property
    def source(self) -> Optional[str]:
        """The source location of the blob as string if known otherwise none.

        If a path is associated with the blob, it will default to the path location.

        Unless explicitly set via a metadata field called "source", in which
        case that value will be used instead.
        """
        if self.metadata and "source" in self.metadata:
            return cast(Optional[str], self.metadata["source"])
        return str(self.path) if self.path else None

    @model_validator(mode='after')
    def check_blob_is_valid(self) -> Self:
        """Verify that either data or path is provided."""
        if not self.data and not self.path:
            raise ValueError("Either data or path must be provided")
        return self

    def is_image(self) -> bool:
        return self.mimetype.startswith("image/") if self.mimetype else False

    def is_pdf(self) -> bool:
        return self.mimetype == "application/pdf" if self.mimetype else False

    def is_audio(self) -> bool:
        return self.mimetype.startswith("audio/") if self.mimetype else False

    def is_video(self) -> bool:
        return self.mimetype.startswith("video/") if self.mimetype else False

    def is_text(self) -> bool:
        app_text_mimetypes = [
            'application/json',
            'application/javascript',
            'application/manifest+json',
            'application/xml',
            'application/x-sh',
            'application/x-python',
        ]
        return (self.mimetype.startswith("text/") or self.mimetype in app_text_mimetypes) if self.mimetype else False

    async def as_bytes(self) -> bytes:
        """Read data as bytes."""
        if self.data is None and (self.path.startswith('http://') or self.path.startswith('https://')):
            response = requests.get(self.path)
            return response.content
        elif self.data is None and is_s3_url(self.path):
            return await read_file_from_s3(self.path)
        elif self.data is None and is_azure_blob_url(self.path):
            return await read_file_from_azure_blob(self.path)
        elif isinstance(self.data, bytes):
            return self.data
        elif isinstance(self.data, str):
            return self.data.encode("utf-8")
        elif self.data is None and self.path:
            with open(str(self.path), "rb") as f:
                return f.read()
        else:
            raise ValueError(f"Unable to get bytes for blob {self}")

    async def as_bytes_io(self) -> BytesIO:
        return BytesIO(await self.as_bytes())

    @classmethod
    def from_path_or_url(
            cls,
            path: Union[str, PurePath],
            *,
            mime_type: Optional[str] = None,
            guess_type: bool = True,
            metadata: Optional[dict] = None,
    ) -> "Blob":
        """Load the blob from a path like object.

        Args:
            path: path like object to file to be read
            mime_type: if provided, will be set as the mime-type of the data
            guess_type: If True, the mimetype will be guessed from the file extension,
                        if a mime-type was not provided
            metadata: Metadata to associate with the blob

        Returns:
            Blob instance
        """
        if mime_type is None and guess_type:
            _mimetype = mimetypes.guess_type(path)[0] if guess_type else None
        else:
            _mimetype = mime_type

        # Convert a path to an absolute path
        if os.path.isfile(path):
            path = os.path.abspath(path)

        # We do not load the data immediately, instead we treat the blob as a
        # reference to the underlying data.
        return cls(
            data=None,
            mimetype=_mimetype,
            path=path,
            metadata=metadata if metadata is not None else {},
        )

    @classmethod
    def from_data(
            cls,
            data: Union[str, bytes],
            *,
            mime_type: str,
            path: Optional[str] = None,
            metadata: Optional[dict] = None,
    ) -> "Blob":
        """Initialize the blob from in-memory data.

        Args:
            data: the in-memory data associated with the blob
            mime_type: if provided, will be set as the mime-type of the data
            path: if provided, will be set as the source from which the data came
            metadata: Metadata to associate with the blob

        Returns:
            Blob instance
        """
        return cls(
            data=data,
            mimetype=mime_type,
            path=path,
            metadata=metadata if metadata is not None else {},
        )

    def __repr__(self) -> str:
        """Define the blob representation."""
        str_repr = f"Blob {id(self)}"
        if self.source:
            str_repr += f" {self.source}"
        return str_repr