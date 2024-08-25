#!/usr/bin/python3
import logging
import traceback
import pathlib
import json
import base64
import hashlib
import locale
from io import BytesIO
from typing import Optional
from datetime import datetime


from PIL import Image
from common import pillow_image_to_base64, expand_pillow_cropped_image

from apis.minio_setup import save_image_to_minio

from analytics_modules.bbox_check_places import check_place
from analytics_modules.persons.persons_milvus_setup import (
    create_schema_master_faces,
    create_schema_cctv_persons_actions,
)
from analytics_modules.persons.persons_roi import ROITimeoutScheduler
from utils.config import genconf, Camera

config = genconf()
import gettext

# gettext.bindtextdomain("cctv_analytics", localedir="locale")
# gettext.textdomain("cctv_analytics")
# _ = gettext.gettext

logging.debug("Using language {}".format(config.language))
lang_to_install = gettext.translation(
    "cctv_analytics",
    localedir=pathlib.Path("cctv_analytics", "locale"),
    languages=[config.language],
)
lang_to_install.install()


def PersonsAnalyticSetup(
    client_milvus,
    deepface_model_embeddings_dimmension,
    ollama_model_embeddings_dimmension,
):
    create_schema_master_faces(client_milvus, deepface_model_embeddings_dimmension)
    create_schema_cctv_persons_actions(
        client_milvus, ollama_model_embeddings_dimmension
    )


class PersonsAnalytic:
    def __init__(
        self,
        camera: Optional[Camera],
        slack_client=None,
        florence_client=None,
        deepface_client=None,
        deepface_model_name=None,
        deepface_detector_name=None,
        milvus_client=None,
        ollama_client=None,
        ollama_embedding_model=None,
        minio_client=None,
        minio_bucket=None,
        setup_roi_schedulers=True,
    ):
        self.camera = camera
        self.slack_client = slack_client
        self.florence_client = florence_client
        self.deepface_client = deepface_client
        self.deepface_model_name = deepface_model_name
        self.deepface_detector_name = deepface_detector_name
        self.milvus_client = milvus_client
        self.camera_name = self.camera.name
        self.ollama_client = ollama_client
        self.ollama_embedding_model = ollama_embedding_model
        self.minio_client = minio_client
        self.minio_bucket = minio_bucket
        if (self.camera.persons_roi) and setup_roi_schedulers:
            self.roi_notification_instances = [
                ROITimeoutScheduler(roi_name, 4 * 60, None, None)
                for roi_name in self.camera.persons_roi
            ]

    def person_save_minio(self, person_image: Image.Image):
        # Save to minio
        try:
            # Create a BytesIO buffer
            img_buffer = BytesIO()
            # Save the image to the buffer in the specified format
            person_image.save(img_buffer, format="JPEG")
            # Get the binary data
            img_binary = img_buffer.getvalue()
            md5_hash = hashlib.md5(img_binary).hexdigest()

            save_image_to_minio(
                self.minio_client,
                img_binary,
                self.minio_bucket,
                f"{self.camera_name}.{md5_hash}",
            )
        except Exception as e:
            logging.error(
                f"[{self.camera_name}][Deepface] Error while saving to minio: {e}"
            )

    def person_get_overlapping_places(self, human_bbox):
        percentage = 0.0425
        # Unpack the crop coordinates
        x1, y1, x2, y2 = human_bbox

        # Calculate the height of the bounding box
        bbox_height = y2 - y1

        # Calculate the height to crop
        crop_height = int(bbox_height * percentage)

        # Calculate the new coordinates for the bottom portion
        new_y1 = y2 - crop_height
        new_y2 = y2

        # Return the new bounding box coordinates
        human_foot_bbox = [x1, new_y1, x2, new_y2]

        # Region of peron be calculated based of person foot bbox!
        human_bbox_overallped_places = []
        if self.camera.places:
            human_bbox_overallped_places = check_place(
                human_foot_bbox, self.camera.places
            )
        return human_bbox_overallped_places

    def person_scan_with_deepface(self, human_cropped_image: Image.Image, formatted_ts):
        """
        Scan a human face with deepface (Embeddings + Facial expression). Save results of scans to milvus DB.
        """
        # Scan with deepface
        try:
            embeddings = self.deepface_client.represent(
                image_raw_b64=pillow_image_to_base64(human_cropped_image),
                model_name=self.deepface_model_name,
                detector_backend=self.deepface_detector_name,
            )
            if embeddings == None:
                logging.debug(f"[{self.camera_name}][Deepface] NOT Found Face")
                return None
            logging.debug(f"[{self.camera_name}][Deepface] Found Face")

            characteristics = self.deepface_client.analyze(
                image_raw_b64=pillow_image_to_base64(human_cropped_image),
                detector_backend=self.deepface_detector_name,
            )
            if characteristics:
                logging.debug(f"[{self.camera_name}][Deepface] Found Characteristics")

            if embeddings != None:
                deepface_results = self.deepface_client.combine_results(
                    embeddings,
                    characteristics,
                )
                if deepface_results:
                    return deepface_results
                return None
        except Exception as e:
            logging.error(
                f"[{self.camera_name}][Deepface] Error response "
                + traceback.format_exc()
            )
            return None

    def save_person_scan_with_deepface(self, deepface_results, formatted_ts):
        for res in deepface_results:
            try:
                data = {
                    "image_source": config.location,
                    "image_id": f"{self.camera_name}",
                    "image_tags": "",
                    "image_facial_area": str(res["facial_area"]),
                    "image_embeddings": res["embedding"],
                    "model_name": self.deepface_model_name,
                    "detector_name": self.deepface_detector_name,
                    "processing_date": formatted_ts,
                }
                if "characteristics" in res:
                    data["image_tags"]: json.loads(res["characteristics"])
                else:
                    data["image_tags"]: {}

                milvus_res = self.milvus_client.insert(
                    collection_name="master_faces",
                    data=[data],
                )
                logging.debug("Successfully inserted face to milvus 'master-faces'")
            except Exception as e:
                logging.error(
                    "Could not insert face to milvus 'master-faces': "
                    + traceback.format_exc()
                )

    def person_roi_slack_notification(
        self,
        expanded_human_cropped_image: Image.Image,
        human_bbox_overallped_places,
        timestamp,
    ):
        # Convert Unix timestamp to datetime object
        dt = datetime.fromtimestamp(timestamp)

        # Format the datetime object
        formatted_ts = dt.strftime("%H:%M %Y-%m-%d (%A)")

        place_roi: Optional[ROITimeoutScheduler] = None

        # CHECK ROI's if defined for camera
        if self.camera.persons_roi and len(human_bbox_overallped_places) > 0:
            logging.info(
                "[ROI]: " + f"Found {len(human_bbox_overallped_places)} places"
            )
            logging.info(
                "[ROI]: "
                + f"Instances setup for this camera - {', '.join([r_ntf_inst.region_name for r_ntf_inst in self.roi_notification_instances])}"
            )
            for place_roi in self.roi_notification_instances:
                for place_overallped in human_bbox_overallped_places:
                    if place_roi.region_name == place_overallped:
                        logging.info(
                            "[ROI]: "
                            + f"Found people in nearby {place_overallped} at {formatted_ts}"
                        )
                        place_roi.callback_start = lambda: self.slack_client.send_slack_media_text_message(
                            [expanded_human_cropped_image],
                            _(
                                f"Found people in nearby {place_overallped} at {formatted_ts}"
                            ),
                        )
                        place_roi.callback_end = lambda: self.slack_client.send_slack_text_message(
                            _(f"Person is no longer visible nearby {place_overallped}"),
                        )
                        place_roi.start()

    def person_caption_action(self, expanded_human_cropped_image: Image.Image):
        """
        Save in Milvus db what `people` at `given time` at `given location` at `given space` are doing.

        Returns caption text
        """
        # Call human to to OD
        res_human_caption = self.florence_client.image_ocr_od_caption(
            base64_image=pillow_image_to_base64(expanded_human_cropped_image),
            prompts=["<DETAILED_CAPTION>"],
        )

        res_dict = res_human_caption[0]
        if res_dict["prompt"] == "<DETAILED_CAPTION>":
            caption = res_dict["res"]
            return caption
        return None

    def save_person_caption_action_to_milvus(
        self, human_bbox_overallped_places, timestamp, caption
    ):

        logging.info(f"[Camera {self.camera_name}] | Person is captioned as: {caption}")

        # Generate embedding
        ollama_embedding_response = self.ollama_client.embeddings(
            model=self.ollama_embedding_model, prompt=caption
        )
        embedding = ollama_embedding_response["embedding"]

        # Save to Milvus
        data = {
            "person_action_caption": caption,
            "embedding_caption_model_name": self.ollama_embedding_model,
            "caption_model_name": "florence-large",
            "person_action_caption_embeddings": embedding,
            "camera_name": self.camera_name,
            "camera_location": config.location,
            "camera_space": ", ".join(human_bbox_overallped_places),
            "processing_timestamp": timestamp,
        }
        milvus_res = self.milvus_client.insert(
            collection_name="cctv_persons_action_captions",
            data=[data],
        )
        logging.debug(
            "Successfully inserted caption of actions taken by human to milvus 'cctv_persons_action_captions'"
        )

    def process_persons(
        self,
        image,
        od_res_dict,
        formatted_ts,
        timestamp,
        send_slack_notification=True,
        save_caption_to_milvus=True,
        save_deepface_to_milvus=True,
    ):
        for index, label in enumerate(od_res_dict["res"]["labels"]):
            if label == "person":
                human_bbox = od_res_dict["res"]["bboxes"][index]
                logging.info(
                    f"[{self.camera_name}][Person Analytics] Found person in labels at [{index}]"
                )
                logging.info(
                    f"[{self.camera_name}][Person Analytics] bbox " + str(human_bbox)
                )

                # Cut Using bbox
                human_cropped_image = image.crop(human_bbox)

                expanded_human_cropped_image = expand_pillow_cropped_image(
                    image, human_bbox
                )

                # Save person cropped image to minio
                self.person_save_minio(human_cropped_image)

                # Grab overlapping places of person
                human_bbox_overallped_places = self.person_get_overlapping_places(
                    human_bbox
                )

                # Send notfications to slack
                if send_slack_notification:  # Caption person action and save in milvus
                    try:
                        self.person_roi_slack_notification(
                            expanded_human_cropped_image,
                            human_bbox_overallped_places,
                            timestamp,
                        )
                    except Exception as e:
                        logging.error("Could not send slack notification" + str(e))

                # Caption person action and save in milvus
                person_caption_action = self.person_caption_action(
                    expanded_human_cropped_image
                )
                if save_caption_to_milvus:
                    self.save_person_caption_action_to_milvus(
                        human_bbox_overallped_places, timestamp, person_caption_action
                    )

                # Deepface scan for faces
                deepface_results = self.person_scan_with_deepface(
                    human_cropped_image, formatted_ts
                )

                if save_deepface_to_milvus and deepface_results:
                    self.save_person_scan_with_deepface(deepface_results, formatted_ts)

                return [
                    human_bbox_overallped_places,
                    person_caption_action,
                    deepface_results,
                ]
