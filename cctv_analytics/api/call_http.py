from flask import Flask, request, jsonify
from datetime import datetime, timedelta
from api.functions import describe_current_cameras


class http_server:
    def __init__(self, host="0.0.0.0", port=8080):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.setup_routes()

    def setup_routes(self):
        @self.app.route("/current", methods=["GET"])
        def current():
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return (
                jsonify(
                    {"current_time": current_time, "res": describe_current_cameras()}
                ),
                200,
            )

    def run(self):
        self.app.run(host=self.host, port=self.port)
