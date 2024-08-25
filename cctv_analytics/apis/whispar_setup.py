import requests
from io import BytesIO
from pydub import AudioSegment
from pydub.silence import split_on_silence
import logging


class WhisparAPI:
    def __init__(self, api_url, api_key) -> None:
        self.api_url = api_url
        self.api_key = api_key

    def transcribe_audio_api(self, audio: BytesIO):
        url = f"{self.api_url}/audio/api/v1/transcriptions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",  # Wstaw tutaj swój API key
            "Content-Type": "multipart/form-data",
        }

        files = {"file": audio, "model": "whisper-1"}

        response = requests.post(url, headers=headers, files=files)

        if response.status_code == 200:
            return response.json()["text"]
        else:
            logging.error("Could not transcribe audio ", response.text)

    def transcribe_whole_segment(self, audio_bytes: BytesIO):
        # Load the MP3 file
        # Testing
        # audio = AudioSegment.from_file("audio.mp3", format="mp3")
        audio = AudioSegment.from_file(audio_bytes, format="wav")

        # Parameters for silence detection
        silence_thresh = -50  # Silence threshold in dB
        min_silence_len = 500  # Minimum length of silence in ms
        keep_silence = 100  # Keep some silence at the beginning and end of each chunk

        # Split the audio based on silence
        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence,
        )
        response = []

        # Minimum chunk length to consider as valid (in milliseconds)
        min_chunk_length = 700

        # Process each chunk and save it
        for i, chunk in enumerate(chunks):
            # Filter out chunks that are too short
            if len(chunk) < min_chunk_length:
                continue

            # Calculate the start time of the current chunk from the beginning of the audio
            start_time = sum(len(chunks[j]) for j in range(i))

            # Export each valid chunk as a separate audio file
            exported_chunk = BytesIO()
            chunk.export(exported_chunk, format="mp3")
            res = self.transcribe_audio_api(exported_chunk)
            if len(res) > 0:
                response.append(
                    {
                        "delta": start_time,
                        "chunk": exported_chunk,
                        "text": res,
                    }
                )
        return response
