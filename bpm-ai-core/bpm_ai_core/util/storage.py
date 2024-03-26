import os
from urllib.parse import urlparse

try:
    from aiobotocore.session import get_session
except ImportError:
    pass

try:
    from azure.storage.blob.aio import BlobClient
except ImportError:
    pass


def is_s3_url(url: str) -> bool:
    return url.startswith('s3://') or (url.startswith('https://') and "amazonaws.com" in url)


async def read_file_from_s3(file_url: str) -> bytes:
    """
    Reads a file from an S3 bucket using a URL-like format.

    Args:
        file_url (str): The URL-like string representing the S3 file location.
                        Formats:
                          - "s3://<bucket-name>/<file-path>"
                          - "https://<bucket-name>.s3.<region>.amazonaws.com/<file-path>"

    Returns:
        bytes: The contents of the file.
    """
    try:
        bucket_name, file_path = await parse_s3_url(file_url)

        async with get_session().create_client('s3') as s3:
            response = await s3.get_object(Bucket=bucket_name, Key=file_path)
            async with response['Body'] as stream:
                return await stream.read()
    except Exception as e:
        raise Exception(f"Error reading file from S3: {str(e)}")


async def parse_s3_url(s3_url: str):
    # Extract bucket name and file path based on the URL format
    parsed_url = urlparse(s3_url)
    if parsed_url.scheme == 's3':
        bucket_name = parsed_url.netloc
        file_path = parsed_url.path.lstrip('/')
    elif parsed_url.scheme == 'https' and '.amazonaws.com' in parsed_url.netloc:
        bucket_name = parsed_url.netloc.split('.')[0]
        file_path = parsed_url.path.lstrip('/')
    else:
        raise ValueError("Invalid S3 file URL format.")
    return bucket_name, file_path


def is_azure_blob_url(url: str) -> bool:
    return "blob.core.windows.net" in url


async def read_file_from_azure_blob(file_url: str) -> bytes:
    """
       Reads a file from Azure Blob Storage using a URL.

       Args:
           file_url (str): The URL of the file in Azure Blob Storage.
                           Format: "https://<account-name>.blob.core.windows.net/<container-name>/<file-path>"

       Returns:
           str: The contents of the file as a string.
       """
    try:
        access_key = os.environ.get('AZURE_STORAGE_ACCESS_KEY')
        async with BlobClient.from_blob_url(file_url, credential=access_key) as blob_client:
            blob = await blob_client.download_blob()
            return await blob.readall()
    except Exception as e:
        raise Exception(f"Error reading file from Azure Blob Storage: {str(e)}")
