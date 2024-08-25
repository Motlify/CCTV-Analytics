import logging
import pathlib

from common import grab_current_images
from utils.config import configure_logging


def run():
    configure_logging()
    tmp_img_dir = pathlib.Path("images")
    tmp_img_dir.mkdir(exist_ok=True)

    for camera in grab_current_images():
        camera["image"].save(
            pathlib.Path(tmp_img_dir, camera["camera"]["name"] + ".jpg")
        )
    logging.info(f"Saved images to {tmp_img_dir}")
