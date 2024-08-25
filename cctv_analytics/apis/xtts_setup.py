from io import BytesIO
import requests
import logging
import json


class XttsAPI:
    def __init__(self, api_url) -> None:
        self.api_url = api_url

    def generate_speaker_embeddings(self, audio: BytesIO):
        url = f"{self.api_url}/clone_speaker"

        files = {"wav_file": audio.getvalue()}
        response = requests.post(url, files=files)

        if response.status_code == 200:
            return response.json()
        else:
            logging.error("Could not transcribe audio ", response.text)
