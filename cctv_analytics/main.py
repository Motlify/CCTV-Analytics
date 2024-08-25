#!/usr/bin/python3
import logging
from utils.config import genconf, configure_logging

configure_logging()

import pytz
from datetime import datetime
import pathlib
import sys
import os
import traceback
import threading
from io import BytesIO
import json
import base64
import hashlib
from typing import Optional
import asyncio

from api.call_slack import slack_socket
from api.call_http import http_server

from analytics_modules.analyze_images import analyze_image
from analytics_modules.analyze_audio import analyze_audio

config = genconf()


def scan_and_detect_images_using_kafka():
    from kafka import KafkaConsumer

    # Create a Kafka consumer instance
    consumer = KafkaConsumer(
        config.kafka.topics.image,
        bootstrap_servers=config.kafka.api_url,
        group_id="CCTV_ANALYTICS_IMAGES_CONSTANT",
        max_poll_records=16,
        enable_auto_commit=True,
        auto_offset_reset="latest",
    )

    # Start consuming messages from the topic
    consumer.subscribe(config.kafka.topics.image)
    logging.info("Starting polling for images using kafka")
    while True:
        records = consumer.poll(timeout_ms=3000)
        consumer.seek_to_end()
        for tp, records_batch in records.items():
            for message in records_batch:
                for camera in config.cameras:
                    try:
                        if camera.name.encode() == message.headers[0][1]:
                            try:
                                analyze_image(message, camera)
                            except Exception as e:
                                logging.warning(
                                    "Could not analyze image" + traceback.print_exc()
                                )
                                try:
                                    slack_client.send_slack_text_message(
                                        "Could not analyze image!"
                                    )
                                except Exception as e:
                                    logging.error("Cannot connect to slack")

                            consumer.commit()
                    except IndexError as e:
                        logging.error("There is no selected headers!" + e)


def scan_and_detect_audio_using_kafka():
    from kafka import KafkaConsumer

    # Create a Kafka consumer instance
    consumer = KafkaConsumer(
        config.kafka.topics.audio,
        bootstrap_servers=config.kafka.api_url,
        group_id="CCTV_ANALYTICS_AUDIO_CONSTANT",
        max_poll_records=16,
        enable_auto_commit=True,
        auto_offset_reset="latest",
    )
    # Start consuming messages from the topic
    consumer.subscribe(config.kafka.topics.audio)
    logging.info("Starting polling for audio using kafka")

    while True:
        records = consumer.poll(timeout_ms=3000)
        consumer.seek_to_end()
        for tp, records_batch in records.items():
            for message in records_batch:
                for camera in config.cameras:
                    try:
                        if camera.name.encode() == message.headers[0][1]:
                            try:
                                analyze_audio(message, camera)
                            except Exception as e:
                                logging.warning(
                                    "Could not analyze audio" + traceback.print_exc()
                                )
                                try:
                                    slack_client.send_slack_text_message(
                                        "Could not analyze audio!"
                                    )
                                except Exception as e:
                                    logging.error("Cannot connect to slack")

                            consumer.commit()
                    except IndexError as e:
                        logging.error("There is no selected headers!" + e)


async def main():
    logging.info("Starting scheduling")
    for func in [
        scan_and_detect_images_using_kafka,
        scan_and_detect_audio_using_kafka,
        slack_socket().run,
        http_server().run,
    ]:
        logging.info("Starting thread" + func.__name__)
        loop = asyncio.get_running_loop()
        awaitable = loop.run_in_executor(None, func)


asyncio.run(main())
