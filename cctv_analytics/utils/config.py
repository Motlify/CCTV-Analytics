from json import loads, dump
import os
import sys
import pathlib
import logging
import json
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError

CONFIG_FILE = pathlib.Path(os.path.dirname(__file__), "config.json")


def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s:[%(funcName)s] %(message)s"
    )
    logging.basicConfig(format="%(asctime)s %(levelname)-8s:[%(funcName)s] %(message)s")
    logging.getLogger("pymilvus").setLevel(logging.CRITICAL)
    logging.getLogger("kafka").setLevel(logging.CRITICAL)
    logging.debug(__name__)


# Configuration file
class SlackConfig(BaseModel):
    oauth_token: str
    app_token: str
    notify_url: str
    channel: str


class DeepfaceConfig(BaseModel):
    detector_name: str
    embedding_dim: int
    deepface_model_name: str
    url: str


class FlorenceConfig(BaseModel):
    url: str


class InfluxDBConfig(BaseModel):
    api_url: str
    bucket: str
    org: str
    token: str


class KafkaTopics(BaseModel):
    image: str
    audio: str


class KafkaConfig(BaseModel):
    api_url: str
    topics: KafkaTopics


class MilvusConfig(BaseModel):
    token: str
    uri: str


class MinIOConfig(BaseModel):
    access_key: str
    bucket_name: str
    bucket_voice: str
    host: str
    location: str
    secret_key: str


class OllamaConfig(BaseModel):
    agent_model: str
    embedding_dim: int
    embedding_model: str
    summarize_model: str
    host: str
    response_language: str


class WhisparConfig(BaseModel):
    api_key: str
    api_url: str


class XTTSConfig(BaseModel):
    api_url: str


class Place(BaseModel):
    name: str = Field(..., description="The name of the place.")
    polygon: List[Tuple[int, int]] = Field(
        None,
        description="The polygon defining the area of the place. If None, the entire camera view is considered the place.",
    )


class Camera(BaseModel):
    name: str = Field(..., description="The name of the camera.")
    places: List[Place] = Field(
        [], description="A list of places defined in this camera's view."
    )
    persons_roi: Optional[List[str]] = Field(
        default=None, description="The ROI for persons to alert with slack."
    )


class Config(BaseModel):
    cameras: List[Camera] = Field(
        [], description="A list of cameras and their configurations."
    )
    deepface: DeepfaceConfig
    florence: FlorenceConfig
    influxdb: InfluxDBConfig
    kafka: KafkaConfig
    milvus: MilvusConfig
    minio: MinIOConfig
    ollama: OllamaConfig
    slack: SlackConfig
    whispar: WhisparConfig
    xtts: XTTSConfig
    language: str = Field(
        ...,
        description="The language to use in user facing interfaces like Slack. Formated in short form as en, pl, de",
    )


def genconf() -> Optional[Config]:
    try:
        configuration = Config.parse_file(CONFIG_FILE)
        return configuration
    except ValidationError as e:
        logging.error(
            f"{CONFIG_FILE} is not a valid Config file, Errors: \n {e.json(indent=4)}"
        )
        sys.exit(1)
    except FileNotFoundError as e:
        logging.error(f"Config File: '{CONFIG_FILE}' could not be found")
        sys.exit(1)
