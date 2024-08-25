from PIL import Image
from io import BytesIO
import base64
import logging
import pathlib
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from io import BytesIO

from utils.config import genconf, Camera

config = genconf()


def pillow_image_to_base64(image: Image.Image):
    """
    Convert the image to a byte array
    """

    buffered = BytesIO()
    image.save(buffered, format="JPEG")  # You can specify the format (JPEG, PNG, etc.)
    img_byte_array = buffered.getvalue()

    # Encode the byte array to base64
    img_base64 = base64.b64encode(img_byte_array).decode("utf-8")
    return img_base64


def expand_pillow_cropped_image(
    image: Image.Image, crop_coordinates, expand_factor=0.15
) -> Image.Image:
    """
    Expands the cropped region by a certain factor in each direction.

    :param image_path: Path to the original image.
    :param crop_coordinates: A tuple of (x1, y1, x2, y2) coordinates for the cropped region.
    :param expand_factor: The factor by which to expand the crop in each direction.
    :return: The expanded cropped image.
    """
    # Load the original image
    # img = Image.open(image_path)
    width, height = image.size

    # Unpack the crop coordinates
    x1, y1, x2, y2 = crop_coordinates

    # Calculate the width and height of the cropped region
    crop_width = x2 - x1
    crop_height = y2 - y1

    # Calculate the expansion size
    expand_w = int(crop_width * expand_factor)
    expand_h = int(crop_height * expand_factor)

    # Calculate new coordinates with the expansion
    new_x1 = max(0, x1 - expand_w)
    new_y1 = max(0, y1 - expand_h)
    new_x2 = min(width, x2 + expand_w)
    new_y2 = min(height, y2 + expand_h)

    # Crop the expanded region from the original image
    expanded_crop = image.crop((new_x1, new_y1, new_x2, new_y2))

    return expanded_crop


class CamerasImages(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    camera: Camera
    image: Image.Image


def check_if_name_exists(list_of_dicts: List[CamerasImages], name_to_check):
    for index, d in enumerate(list_of_dicts):
        if d.camera.name == name_to_check:
            return True, f"'{name_to_check}' exists in the dictionary at index {index}."
    return False, f"'{name_to_check}' does not exist in any dictionary."


def grab_current_images() -> List[CamerasImages]:
    from kafka import KafkaConsumer

    all_cameras_images: List[CamerasImages] = []
    # Create a Kafka consumer instance
    consumer = KafkaConsumer(
        config.kafka.topics.image,
        bootstrap_servers=config.kafka.api_url,
        group_id="CCTV_ANALYTICS_API",
        max_poll_records=16,
        enable_auto_commit=True,
        auto_offset_reset="latest",
    )

    # Start consuming messages from the topic
    matched_all = False
    consumer.subscribe(config.kafka.topics.image)
    logging.info("Starting polling for images using kafka")
    logging.info(
        "Scrapping image from cameras: "
        + ", ".join([cam.name for cam in config.cameras])
    )
    while True and not matched_all:
        records = consumer.poll(timeout_ms=3000)
        consumer.seek_to_end()
        for tp, records_batch in records.items():
            for message in records_batch:
                for camera in config.cameras:
                    if camera.name.encode() == message.headers[0][1]:
                        ts_parsed = str(message.headers[1][1].decode()).split(".")[0]
                        timestamp = int(ts_parsed)
                        image_bytes = message.value

                        # Open the image using PIL
                        image_buffer = BytesIO(image_bytes)
                        if check_if_name_exists(all_cameras_images, camera.name):
                            logging.info(f"Getting new camera feed {camera.name}")
                            camera_info = CamerasImages(
                                image=Image.open(image_buffer), camera=camera
                            )
                            all_cameras_images.append(camera_info)
                    if len(all_cameras_images) == len(config.cameras):
                        matched_all = True
    return all_cameras_images
