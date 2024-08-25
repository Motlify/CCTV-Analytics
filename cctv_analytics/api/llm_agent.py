from utils.config import genconf
import requests
import json
import traceback
import logging

from api.functions import describe_current_cameras

config = genconf()

# Ollama client
ollama_url = config.ollama.host
ollama_agent_model = config.ollama.agent_model

SYSTEM = """
## TASK
You are cctv agent that has access ONLY to camera feeds. 
You will respond in JSON.
## RULES
Never respond to other requests.
Do not make general assumptions. 
Keep names of cameras
## RESPONSE SCHEMA
{
  "type": "function | message",
  "content": "string",
  "params": [param1, param2]
}
## COMMANDS
If user will ask about current situation of cctv you will respond with function 'describe_current_cameras'.
If user will ask about situations from few hours backs of cctv you will respond with function 'describe_cameras_back_in_time' with param hours=<user projected hours, if not given set to None>.
"""


def ask_assistant(user_message):
    data = {
        "model": ollama_agent_model,
        "prompt": SYSTEM + "\n ## User Message \n" + user_message,
        "temperature": 0.5,
        "format": "json",
        "stream": False,
    }
    response = requests.post(
        ollama_url + "/api/generate",
        headers={"Content-Type": "application/json"},
        json=data,
    )
    if response.status_code == 200:
        result = response.json()
        try:
            raw_chain = json.loads(result["response"])
            return raw_chain
        except json.JSONDecodeError as e:
            return llm_generate_chain(task)
    else:
        logging.error("Request failed with status code:", response.status_code)


def dispatch_llm(user_message):
    try:
        res = ask_assistant(user_message)
    except Exception as e:
        return "Something went wrong"
    try:
        if res["type"] == "function":
            if res["content"] == "describe_current_cameras":
                return describe_current_cameras(user_message)
        if res["type"] == "message":
            return res["content"]
    except Exception as e:
        logging.error(traceback.format_exc())
        return "Something went wrong"
