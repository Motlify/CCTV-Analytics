import requests
import logging
import slack_sdk
from typing import List, Optional
from utils.config import SlackConfig
from PIL import Image
from io import BytesIO


class SlackAPI:
    def __init__(self, slack_config: Optional[SlackConfig] = None):
        self.config = slack_config
        self.client = slack_sdk.WebClient(token=self.config.oauth_token)

    def send_slack_media_text_message(self, images_list: List[Image.Image], message):
        logging.debug(
            f"[SLACK] Sending text with media files ({len(images_list)}) and message: '{message}'"
        )
        file_uploads = []

        for idx, image in enumerate(images_list):
            # Create a BytesIO object
            image_bytes = BytesIO()

            # Save the image to the BytesIO object in a specific format (e.g., JPEG)
            image.save(image_bytes, format="JPEG")

            # Get the byte data from the BytesIO object
            image_data = image_bytes.getvalue()
            file_uploads.append(
                {"file": image_data, "filename": f"image_{idx}.jpeg", "title": message}
            )

        new_file = self.client.files_upload_v2(
            file_uploads=file_uploads,
            channel=self.config.channel,
            initial_comment=message,
        )

    def send_slack_text_message(self, message):
        logging.debug(f"[SLACK] Sending text with message: '{message}'")
        try:
            url = self.config.notify_url
            response = requests.post(
                url,
                headers={"Content-type": "application/json"},
                json={"text": message},
            )
            if response.ok:
                logging.debug("Message sent successfully.")
            else:
                logging.error("Failed to send the message:", response.text)
        except Exception as e:
            logging.error("Could not send slack message" + e)
