from io import BytesIO
import traceback
from datetime import datetime
import pytz
import logging

from pymilvus import MilvusClient
from ollama import Client as OllamaClient

from apis.whispar_setup import WhisparAPI
from apis.minio_setup import setup_minio, save_image_to_minio
from apis.slack_setup import SlackAPI

from apis.minio_setup import save_voice_to_minio
from apis.xtts_setup import XttsAPI
from analytics_modules.audio.audio_milvus_setup import (
    create_schema_cctv_transcription,
    create_schema_voice_embeddings,
)


from utils.config import genconf

config = genconf()


# Minio Client
minio_bucket = config.minio.bucket_name
minio_client = setup_minio(
    config.minio.host,
    config.minio.access_key,
    config.minio.secret_key,
    config.minio.location,
    minio_bucket,
)


# Milvus configuration
milvus_client = MilvusClient(uri=config.milvus.uri, token=config.milvus.token)

# Whispar configuration
whispar_client = WhisparAPI(config.whispar.api_url, config.whispar.api_key)

# XTTS configuration
xtts_client = XttsAPI(config.xtts.api_url)
xtts_voice_embedding_dimmension = 512

# Slack notifications
slack_client = SlackAPI(config.slack)

# OLLAMA - Person caption embedding model
ollama_embedding_model = config.ollama.embedding_model
ollama_model_embeddings_dimmension = config.ollama.embedding_dim
ollama_client = OllamaClient(host=config.ollama.host)

create_schema_cctv_transcription(milvus_client, ollama_model_embeddings_dimmension)
create_schema_voice_embeddings(milvus_client, xtts_voice_embedding_dimmension)


def save_voice_chunks_to_minio(res_transcriptions_list, timestamp, camera_name):
    for chunk_info in res_transcriptions_list:
        save_voice_to_minio(
            minio_client,
            chunk_info["chunk"],
            minio_bucket,
            f"{camera_name}.{timestamp}.wav",
        )


def save_audio_transcription(res_transcriptions_list, timestamp, camera_name):
    transcription_text = ""
    for res_dict in res_transcriptions_list:
        transcription_text += res_dict["text"] + " "

    if len(res_transcriptions_list) > 0:
        # Generate embeddings for text
        ollama_embedding_response = ollama_client.embeddings(
            model=ollama_embedding_model, prompt=transcription_text
        )
        text_embedding = ollama_embedding_response["embedding"]

        # Save to Milvus
        data = {
            "transcription_text": transcription_text,
            "transcription_text_embeddings": text_embedding,
            "embedding_transcription_model_name": ollama_embedding_model,
            "camera_name": camera_name,
            "camera_location": config.location,
            "processing_timestamp": timestamp,
        }

        milvus_res = milvus_client.insert(
            collection_name="cctv_transcription",
            data=[data],
        )
        logging.debug(
            "Successfully inserted caption from audio to milvus 'cctv_transcription'"
        )


def save_multiple_voice_embeddings_to_milvus(
    res_transcriptions_list, timestamp, camera_name
):
    # Split audio by delta - list might be empty!
    for chunk_info in res_transcriptions_list:
        res = xtts_client.generate_speaker_embeddings(chunk_info["chunk"])
        # Save to Milvus
        data = {
            "speaker_embedding": res["speaker_embedding"],
            "embedding_voice_model_name": "COQUI-XTTS",
            "xtts_voice_embedding_dimmension": xtts_voice_embedding_dimmension,
            "camera_name": camera_name,
            "camera_location": config.location,
            "processing_timestamp": timestamp,
        }
        milvus_res = milvus_client.insert(
            collection_name="master_voices",
            data=[data],
        )
        logging.debug(
            "Successfully saved voice embeddings from audio to milvus 'master-voices'"
        )


def analyze_audio(message, camera):
    try:
        camera_name = camera.name
        ts_parsed = str(message.headers[1][1].decode()).split(".")[0]
        timestamp = int(ts_parsed)

        # Capture data from CCTV
        audio = message.value

        audio_buffer = BytesIO(audio)

        res_transcriptions_list = whispar_client.transcribe_whole_segment(audio_buffer)

        save_audio_transcription(res_transcriptions_list, timestamp, camera_name)

        save_multiple_voice_embeddings_to_milvus(
            res_transcriptions_list, timestamp, camera_name
        )

        save_voice_chunks_to_minio(res_transcriptions_list, timestamp, camera_name)

    except Exception as e:
        logging.error(traceback.print_exc())
