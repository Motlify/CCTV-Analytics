import requests
import base64
import json
import logging


def encode_image(image_path_l):
    with open(image_path_l, "rb") as imagefile:
        convert = base64.b64encode(imagefile.read()).decode("utf-8")
        return convert


class DeepfaceAPI:
    def __init__(self, url):
        self._url = url

    def represent(
        self, image_raw_b64, model_name="Facenet", detector_backend="mtcnn"
    ) -> list:
        imageb64 = "data:image/jpeg;base64," + str(image_raw_b64)
        body = {
            "img_path": imageb64,
            "model_name": model_name,
            "detector_backend": detector_backend,
        }
        res = requests.post(f"{self._url}/represent", json=body)
        if res.status_code == 200:
            try:
                logging.debug(res)
                data = res.json()
                return data["results"]
            except Exception as e:
                return None
        return None

    def analyze(self, image_raw_b64, detector_backend="mtcnn") -> list:
        imageb64 = "data:image/jpeg;base64," + str(image_raw_b64)
        body = {
            "img_path": imageb64,
            "detector_backend": detector_backend,
            "actions": ["age", "gender", "emotion", "race"],
        }
        res = requests.post(f"{self._url}/analyze", json=body)
        if res.status_code == 200:
            try:
                logging.debug(res)
                data = res.json()
                logging.debug(data)
                return data["results"]
            except Exception as e:
                return None
        return None

    def combine_results(self, embeddings, characteristics):
        results = []
        for e in embeddings:
            results.append(
                {"embedding": e["embedding"], "facial_area": e["facial_area"]}
            )
        if characteristics is not None:
            for i, c in enumerate(characteristics):
                valuable_characteristics = [
                    "age",
                    "dominant_emotion",
                    "dominant_gender",
                    "dominant_race",
                ]
                try:
                    results[i]["characteristics"] = {}
                    for v in valuable_characteristics:
                        results[i]["characteristics"][v] = c[v]
                except Exception as e:
                    logging.warning("Could not get characteristics")
        return results
