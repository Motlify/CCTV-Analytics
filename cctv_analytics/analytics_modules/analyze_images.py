import traceback
import logging
from io import BytesIO
from PIL import Image
import base64
import json
from datetime import datetime
import pytz
from ollama import Client as OllamaClient
from pymilvus import MilvusClient


from apis.deepface_setup import DeepfaceAPI, encode_image
from apis.florence_setup import FlorenceAPI
from apis.minio_setup import setup_minio, save_image_to_minio
from apis.slack_setup import SlackAPI
from apis.influxdb_setup import InfluxAPI

from analytics_modules.car_plates.car_plates_analytics import CarPlatesAnalytic
from analytics_modules.persons.persons_analytics import (
    PersonsAnalytic,
    PersonsAnalyticSetup,
)

from utils.config import genconf

config = genconf()


# InfluxDB
influx_client = InfluxAPI(
    url=config.influxdb.api_url,
    token=config.influxdb.token,
    org=config.influxdb.org,
    bucket=config.influxdb.bucket,
)

# Minio Client
minio_bucket = config.minio.bucket_name
minio_client = setup_minio(
    config.minio.host,
    config.minio.access_key,
    config.minio.secret_key,
    config.minio.location,
    minio_bucket,
)

# Deepface configuration
deepface_client = DeepfaceAPI(config.deepface.url)
deepface_model_name = config.deepface.deepface_model_name
deepface_detector_name = config.deepface.deepface_model_name
deepface_model_embeddings_dimmension = config.deepface.embedding_dim

# OLLAMA - Person caption embedding model
ollama_embedding_model = config.ollama.embedding_model
ollama_model_embeddings_dimmension = config.ollama.embedding_dim
ollama_url = config.ollama.host
ollama_client = OllamaClient(host=ollama_url)

# Milvus configuration
milvus_client = MilvusClient(uri=config.milvus.uri, token=config.milvus.token)

# Florence configuration
florence_client = FlorenceAPI(config.florence.url)

# Slack notifications
slack_client = SlackAPI(config.slack)

# Setup Persons Analytics
PersonsAnalyticSetup(
    milvus_client,
    deepface_model_embeddings_dimmension,
    ollama_model_embeddings_dimmension,
)

persons_analytics_instances = [
    PersonsAnalytic(
        camera,
        slack_client,
        florence_client,
        deepface_client,
        deepface_model_name,
        deepface_detector_name,
        milvus_client,
        ollama_client,
        ollama_embedding_model,
        minio_client,
        minio_bucket,
    )
    for camera in config.cameras
]


def analyze_image(message, camera):
    try:
        ts_parsed = str(message.headers[1][1].decode()).split(".")[0]
        timestamp = int(ts_parsed)

        # Convert the timestamp to a datetime object
        dt = datetime.fromtimestamp(timestamp, pytz.timezone("Europe/Warsaw"))
        formatted_ts = dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")

        # Capture data from CCTV
        image_bytes = message.value

        # Open the image using PIL
        image_buffer = BytesIO(image_bytes)
        image = Image.open(image_buffer)

        # Encode the image buffer to Base64
        image_buffer.seek(0)  # Make sure the buffer is at the beginning
        img_base64 = base64.b64encode(image_buffer.getvalue()).decode("utf-8")

        # Detect objects
        od_rest_dict_list = florence_client.image_ocr_od_caption(
            base64_image=img_base64, prompts=["<OD>"]
        )

        for od_res_dict in od_rest_dict_list:
            milvus_cctv_data = {}
            if od_res_dict["prompt"] == "<OD>":
                od_res_dict["res"] = json.loads(od_res_dict["res"].replace("'", '"'))
                logging.debug(
                    f"[Camera {camera.name}] | {formatted_ts} | found objects: "
                    + ", ".join(od_res_dict["res"]["labels"])
                )
                # Person Analytics
                try:
                    found_inst_p_a: Optional[PersonsAnalytic] = None  # Type annotation
                    found_inst_p_a = next(
                        (
                            inst
                            for inst in persons_analytics_instances
                            if inst.camera_name == camera.name
                        ),
                        None,
                    )
                    found_inst_p_a.process_persons(
                        image, od_res_dict, formatted_ts, timestamp
                    )
                except Exception as e:
                    logging.error(
                        "Could not process persons analytics" + traceback.format_exc()
                    )

                # Car plates Analytics
                try:
                    if "car_plates" in camera:
                        CarPlatesAnalytic(
                            camera, florence_client, influx_client
                        ).process_cars(image, od_res_dict, timestamp)
                except Exception as e:
                    logging.error("Could not process car analytics")

    except Exception as e:
        logging.error(traceback.print_exc())
