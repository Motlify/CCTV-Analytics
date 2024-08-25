from minio import Minio
from minio.error import S3Error
from PIL import Image
from io import BytesIO
import logging


def setup_minio(host, access_key, secret_key, minio_location, bucket_name):
    client = Minio(
        host,
        access_key=access_key,
        secret_key=secret_key,
        secure=False,
    )

    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logging.debug(f"Bucket '{bucket_name}' created successfully.")
        else:
            logging.debug(f"Bucket '{bucket_name}' already exists.")
    except S3Error as e:
        logging.error(f"Error occurred: {e}")
    return client


def save_image_to_minio(client, img_binary, bucket_name: str, object_id: str):
    result = client.put_object(
        bucket_name,
        object_id,
        BytesIO(img_binary),
        len(img_binary),
        content_type="image/jpeg",
    )
    return result.object_name


def save_voice_to_minio(
    client, voice_bytesio: BytesIO, bucket_name: str, object_id: str
):
    result = client.put_object(
        bucket_name,
        object_id,
        voice_bytesio,
        len(voice_bytesio.getvalue()),
        content_type="audio/wav",
    )
    return result.object_name
