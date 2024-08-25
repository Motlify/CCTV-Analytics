from datetime import datetime
from typing import Optional
from common import pillow_image_to_base64
from PIL import Image
import logging

from influxdb_client import InfluxDBClient

from apis.florence_setup import FlorenceAPI
from apis.influxdb_setup import InfluxAPI


class CarPlatesAnalytic:
    def __init__(self, camera, florence_client, influx_client):
        self.camera = camera
        self.florence_client: Optional[FlorenceAPI] = florence_client
        self.influx_client: Optional[InfluxAPI] = influx_client
        self.camera_name = self.camera["name"]

    def _run_ocr_over_car(self, image, dt_time):
        res = self.florence_client.image_ocr_od_caption(
            base64_image=pillow_image_to_base64(image), prompts=["<OCR>"]
        )
        if len(res) > 0:
            plate_text = res[0]["res"]
            p = (
                self.influx_client.client.Point("plates_detected")
                .tag("camera", self.camera_name)
                .field("confidence", 1)
                .field("text", plate_text)
                .field("time", dt_time)
            )

    def process_cars(self, image, od_res_dict, timestamp):
        for index, label in enumerate(od_res_dict["res"]["labels"]):
            if label in ["car", "van", "vehicle registration plate", "truck"]:
                car_bbox = od_res_dict["res"]["bboxes"][index]
                logging.info(
                    f"[{self.camera_name}][Car Plates] Found car in labels at [{index}]"
                )
                logging.info(f"[{self.camera_name}][Car Plates] bbox " + str(car_bbox))

                # Cut Using bbox
                car_cropped_image = image.crop(car_bbox)

                # Run OCR and save to influxdb
                self._run_ocr_over_car(
                    car_cropped_image, datetime.fromtimestamp(timestamp)
                )
