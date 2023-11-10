import boto3
import os
from pathlib import Path
from io import BytesIO
from typing import Tuple, BinaryIO, Any
import pandas as pd


def create_s3_client(service_account_key_id: str, service_account_secret: str) -> Any:
    return boto3.client(
        "s3",
        endpoint_url="https://storage.yandexcloud.net",
        aws_access_key_id=service_account_key_id,
        aws_secret_access_key=service_account_secret,
    )


def upload_to_yandex_storage(
        file: BinaryIO,
        original_filename: str,
        s3: Any,
        bucket_name: str,
        file_ttl: int,
) -> Tuple[str, str]:
    try:
        object_name = f"{os.urandom(8).hex()}_{Path(original_filename)}"
        s3.upload_fileobj(file, bucket_name, object_name)
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_name},
            ExpiresIn=file_ttl,
        )
        return url, object_name
    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}")