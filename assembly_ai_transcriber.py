import argparse
import os
import uuid
import tempfile
import requests
import time
import logging

logger = logging.getLogger(__name__)

class AssemblyAITranscriber:
    upload_endpoint = "https://api.assemblyai.com/v2/upload"
    transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
    temp_dir = "./temp_audio"

    def __init__(self, api_key):
        self.api_key = api_key
        self.header = {
            'authorization': self.api_key,
            'content-type': 'application/json'
        }

    def _read_file(self, filename, chunk_size=5242880):
        with open(filename, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                yield data

    def _upload_file(self, audio_file):
        try:
            upload_response = requests.post(
                self.upload_endpoint,
                headers=self.header, data=self._read_file(audio_file)
            )
            upload_response.raise_for_status()  # Raises a HTTPError if the response was an error
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to upload file: {e}")
            raise
        else:
            upload_response_json = upload_response.json()
            logger.info(f"Successfully uploaded file. Response: {upload_response_json}")
            return upload_response_json

    def _request_transcript(self, upload_url):
        transcript_request = {
            'audio_url': upload_url['upload_url']
        }
        transcript_response = requests.post(
            self.transcript_endpoint,
            json=transcript_request,
            headers=self.header
        )
        return transcript_response.json()

    def _make_polling_endpoint(self, transcript_response):
        polling_endpoint = "https://api.assemblyai.com/v2/transcript/"
        polling_endpoint += transcript_response['id']
        return polling_endpoint

    def _wait_for_completion(self, polling_endpoint):
        while True:
            polling_response = requests.get(polling_endpoint, headers=self.header)
            polling_response = polling_response.json()

            if polling_response['status'] == 'completed':
                break

            time.sleep(5)

    def _get_paragraphs(self, polling_endpoint):
        paragraphs_response = requests.get(polling_endpoint + "/paragraphs", headers=self.header)
        paragraphs_response = paragraphs_response.json()

        paragraphs = []
        for para in paragraphs_response['paragraphs']:
            paragraphs.append(para)

        return paragraphs

    def transcribe_audio(self, audio_file):
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        with tempfile.NamedTemporaryFile(dir=self.temp_dir, delete=False) as temp_file:
            audio_file.save(temp_file.name)

        try:
            upload_url = self._upload_file(temp_file.name)
            transcript_response = self._request_transcript(upload_url)
            polling_endpoint = self._make_polling_endpoint(transcript_response)
            self._wait_for_completion(polling_endpoint)
            paragraphs = self._get_paragraphs(polling_endpoint)
        finally:
            try:
                os.remove(temp_file.name)
            except OSError as e:
                print(f"Error deleting temp file: {e}")

        transcript = []
        for para in paragraphs:
            transcript.append(para['text'])
        
        return "\n".join(transcript)
