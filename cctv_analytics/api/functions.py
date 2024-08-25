import logging
from apis.florence_setup import FlorenceAPI
from apis.deepface_setup import DeepfaceAPI
from ollama import Client as OllamaClient
import traceback

from analytics_modules.persons.persons_analytics import PersonsAnalytic
from common import (
    pillow_image_to_base64,
    expand_pillow_cropped_image,
    grab_current_images,
)
from io import BytesIO
from PIL import Image
import json
from utils.config import genconf

config = genconf()

# Florence configuration
florence_client = FlorenceAPI(config.florence.url)

# Deepface configuration
deepface_client = DeepfaceAPI(config.deepface.url)
deepface_model_name = config.deepface.deepface_model_name
deepface_detector_name = config.deepface.detector_name
deepface_model_embeddings_dimmension = config.deepface.embedding_dim

# Ollama client
ollama_summarize_model = config.ollama.summarize_model
ollama_client = OllamaClient(host=config.ollama.host)


def describe_current_cameras(additional_user_context="") -> str:
    all_cameras_images = grab_current_images()
    all_prompts = ""
    human_text_res = ""
    general_text_res = ""
    for image_data in all_cameras_images:
        od_rest_dict_list = florence_client.image_ocr_od_caption(
            base64_image=pillow_image_to_base64(image_data.image),
            prompts=["<OD>", "<DETAILED_CAPTION>"],
        )
        for od_res_dict in od_rest_dict_list:
            if od_res_dict["prompt"] == "<OD>":
                od_res_dict["res"] = json.loads(od_res_dict["res"].replace("'", '"'))
                for index, label in enumerate(od_res_dict["res"]["labels"]):
                    if label == "person":
                        try:
                            human_bbox = od_res_dict[0]["res"]["bboxes"][index]
                            pa = PersonsAnalytic(
                                camera,
                                None,
                                florence_client,
                                deepface_client,
                                deepface_model_name,
                                deepface_detector_name,
                                None,
                                None,
                                None,
                                None,
                                None,
                            )
                            places = pa.person_get_overlapping_places(human_bbox)
                            caption = pa.person_caption_action(
                                expand_pillow_cropped_image(
                                    image_data.image, human_bbox
                                )
                            )

                            human_text_res += f"{caption}. The places of this action took place at: {places} \n\n"
                        except Exception as e:
                            logging.error("Could not detect human")
            if od_res_dict["prompt"] == "<DETAILED_CAPTION>":
                general_text_res += (
                    f'{od_res_dict["res"]} It happend at {image_data.camera.name}'
                )
    all_prompts = "### General\n " + general_text_res
    if len(human_text_res) > 0:
        all_prompts += "### Human\n " + human_text_res
    if len(all_prompts) == 0:
        return "No human was detected."
    else:
        response = ollama_client.chat(
            model=ollama_summarize_model,
            options={
                "temperature": 0.05,
            },
            messages=[
                {
                    "role": "user",
                    "content": f"## Use {config.ollama.response_language} language in response. \n Task\n Summarize feeds of cameras \n ## Feeds\n"
                    + all_prompts,
                },
            ],
        )
        return response["message"]["content"]
