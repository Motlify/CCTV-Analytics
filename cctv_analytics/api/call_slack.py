import os
import re

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler


from api.llm_agent import dispatch_llm
from utils.config import genconf

config = genconf()


def markdown_to_mrkdwn(markdown_text):
    # Convert unordered lists
    markdown_text = re.sub(r"^- (.*)", r"• \1", markdown_text, flags=re.MULTILINE)

    # Convert bold text
    markdown_text = re.sub(r"\*\*(.*?)\*\*", r"*\1*", markdown_text)

    # Convert italic text
    markdown_text = re.sub(r"_(.*?)_", r"_\1_", markdown_text)

    # Convert headings
    markdown_text = re.sub(r"###### (.*)", r"_\1_", markdown_text)
    markdown_text = re.sub(r"##### (.*)", r"_\1_", markdown_text)
    markdown_text = re.sub(r"#### (.*)", r"_\1_", markdown_text)
    markdown_text = re.sub(r"### (.*)", r"_\1_", markdown_text)
    markdown_text = re.sub(r"## (.*)", r"*\1*", markdown_text)
    markdown_text = re.sub(r"# (.*)", r"*\1*", markdown_text)

    return markdown_text


class slack_socket:
    def __init__(self):
        # Install the Slack app and get xoxb- token in advance
        self.app = App(token=config.slack.oauth_token)
        self.setup_routes()

    def setup_routes(self):
        @self.app.event("message")
        def handle_message_events(body, say):
            if "subtype" in body["event"]:
                if body["event"]["subtype"] == "bot_message":
                    return
            else:
                res = dispatch_llm(body["event"]["text"])
                say(
                    blocks=[
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": markdown_to_mrkdwn(res),
                            },
                        }
                    ],
                    text=res,
                )

    def run(self):
        SocketModeHandler(self.app, config.slack.app_token).start()
