import json
import os
from dotenv import load_dotenv
import numpy as np
import requests
import streamlit as st

from utils.s3_helpers import create_s3_client, upload_to_yandex_storage


load_dotenv()


def process_video(video_url: str, server_url: str, url_type: str):
    data = {"video_name": video_url,
            "url_type": url_type}
    return requests.post(
        server_url, json=data, timeout=8000
    )


def main():
    st.set_page_config(page_title="deepfake_detection", page_icon="src/interface/logo.png")
    st.title("Deepfake Detector")

    service_account_key_id = os.environ.get("SERVICE_ACCOUNT_KEY_ID")
    service_account_secret = os.environ.get("SERVICE_ACCOUNT_SECRET")
    server_url = "http://src-server-1:8000/deepfake_predict"
    bucket_name = "pronomuos"
    file_ttl = 1800

    if service_account_key_id is None or service_account_secret is None:
        raise EnvironmentError(
            "SERVICE_ACCOUNT_KEY_ID and SERVICE_ACCOUNT_SECRET must be set"
        )

    s3 = create_s3_client(service_account_key_id, service_account_secret)

    with st.form("Uploaded video"):
        uploaded_video = st.file_uploader(
            "Choose a video...",
            type=["mp4", "mpeg"],
            accept_multiple_files=False,
        )
        submit_button = st.form_submit_button("Process video")

        object_name = None
        if uploaded_video is not None:
            try:
                video_url, object_name = upload_to_yandex_storage(
                    uploaded_video, uploaded_video.name, s3, bucket_name, file_ttl
                )
            except RuntimeError as e:
                st.write(str(e))

        if submit_button:
            if object_name is not None:
                with st.spinner(f"Processing video..."):
                    response = process_video(object_name, server_url, url_type="uploaded")
                    st.write(f"The video has deepfakes - {response.headers['is_deepfake']}")
                    if response.headers['is_deepfake'] == "True":
                        st.write("Frame with deepfake is shown below")
                        st.image(response.content)

    with st.form("Youtube video link"):

        youtube_url = st.text_input(f"YouTube video link")
        submit_button = st.form_submit_button("Process video")

        if submit_button:
            if youtube_url is not None:
                with st.spinner(f"Processing video..."):
                    response = process_video(youtube_url, server_url, url_type="youtube")
                    st.write(f"The video has deepfakes - {response.headers['is_deepfake']}")
                    if response.headers['is_deepfake'] == "True":
                        st.write("Frame with deepfake is shown below")
                        st.image(response.content)


if __name__ == "__main__":
    main()
