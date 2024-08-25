import requests
import base64
import logging


class FlorenceAPI:
    def __init__(self, api_url: str):
        self.api_url = api_url

    def image_analytics_per_prompt(self, base64_image, prompt):
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer none"}

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 300,
        }

        response = requests.post(
            f"{self.api_url}/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            logging.warning(
                "Could not analyze image"
                + str(response.status_code)
                + " | "
                + response.text
            )
        return None

    def image_ocr_od_caption(self, image_path=None, base64_image=None, prompts=[]):
        """
        prompts: List from ["<OCR>", "<OD>", "<DETAILED_CAPTION>"]
        """
        if prompts == []:
            raise ValueError("No prompts provided.")

        def encode_image(image_path_l):
            with open(image_path_l, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        if image_path:
            # Getting the base64 string
            base64_image = encode_image(image_path)
        if base64_image:
            res = []
            for prompt in prompts:
                curr_res = self.image_analytics_per_prompt(base64_image, prompt)
                if curr_res is not None:
                    res.append(
                        {
                            "prompt": prompt,
                            "res": curr_res,
                        }
                    )
            return res
