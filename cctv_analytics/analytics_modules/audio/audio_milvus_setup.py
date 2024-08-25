import time
from pymilvus import DataType
import logging


def create_schema_cctv_transcription(client_milvus, ollama_model_embeddings_dimmension):
    collection_name = "cctv_transcription"
    exists = client_milvus.has_collection(collection_name)

    if exists:
        logging.info("Collection " + collection_name + " exists")
    else:
        schema = client_milvus.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )

        schema.add_field(field_name="index", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="camera_name", datatype=DataType.VARCHAR, max_length=1024
        )
        schema.add_field(
            field_name="camera_location", datatype=DataType.VARCHAR, max_length=1024
        )
        schema.add_field(
            field_name="transcription_text", datatype=DataType.VARCHAR, max_length=8192
        )
        schema.add_field(
            field_name="embedding_transcription_model_name",
            datatype=DataType.VARCHAR,
            max_length=128,
        )
        schema.add_field(
            field_name="transcription_text_embeddings",
            datatype=DataType.FLOAT_VECTOR,
            dim=ollama_model_embeddings_dimmension,
        )
        schema.add_field(
            field_name="processing_timestamp", datatype=DataType.INT64, max_length=128
        )

        index_params = client_milvus.prepare_index_params()
        index_params.add_index(field_name="index", index_type="STL_SORT")
        index_params.add_index(
            field_name="transcription_text_embeddings",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 128},
        )
        client_milvus.create_collection(
            collection_name=collection_name, schema=schema, index_params=index_params
        )

        time.sleep(5)

        res = client_milvus.get_load_state(collection_name=collection_name)

        logging.info(res)


def create_schema_voice_embeddings(client_milvus, xtts_voice_embedding_dimmension):
    collection_name = "master_voices"

    exists = client_milvus.has_collection(collection_name)

    if exists:
        logging.info("Collection " + collection_name + " exists")
    else:
        schema = client_milvus.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )

        schema.add_field(field_name="index", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="camera_name", datatype=DataType.VARCHAR, max_length=1024
        )
        schema.add_field(
            field_name="camera_location", datatype=DataType.VARCHAR, max_length=1024
        )
        schema.add_field(
            field_name="embedding_voice_model_name",
            datatype=DataType.VARCHAR,
            max_length=128,
        )
        schema.add_field(
            field_name="speaker_embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=xtts_voice_embedding_dimmension,
        )
        schema.add_field(
            field_name="processing_timestamp", datatype=DataType.INT64, max_length=128
        )

        index_params = client_milvus.prepare_index_params()
        index_params.add_index(field_name="index", index_type="STL_SORT")
        index_params.add_index(
            field_name="speaker_embedding",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 128},
        )
        client_milvus.create_collection(
            collection_name=collection_name, schema=schema, index_params=index_params
        )

        time.sleep(5)

        res = client_milvus.get_load_state(collection_name=collection_name)

        logging.info(res)
