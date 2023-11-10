import io
import json
import os
import re

import numpy as np
import torch

import uvicorn
from fastapi import FastAPI, APIRouter
from fastapi.openapi.utils import get_openapi
from starlette.requests import Request
from starlette.responses import Response

from classifiers import DeepFakeClassifier
from kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video
from client.utils.s3_helpers import create_s3_client
from imageio import v3 as iio
from pytube import YouTube


app = FastAPI()
router = APIRouter()

###################
# APIs
###################

models = []
model_paths = [os.path.join("../model", "weights", "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36")]
for path in model_paths:
    model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cuda")
    print("loading state dict {}".format(path))
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
    model.eval()
    del checkpoint
    models.append(model.half())

frames_per_video = 32
video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
service_account_key_id = os.environ.get("SERVICE_ACCOUNT_KEY_ID")
service_account_secret = os.environ.get("SERVICE_ACCOUNT_SECRET")
server_url = "http://localhost:8080/deepfake_predict"
bucket_name = "pronomuos"
file_ttl = 1800

data_dir = "../data"
os.makedirs(os.path.join(data_dir), exist_ok=True)

if service_account_key_id is None or service_account_secret is None:
    raise EnvironmentError(
        "SERVICE_ACCOUNT_KEY_ID and SERVICE_ACCOUNT_SECRET must be set"
    )

s3 = create_s3_client(service_account_key_id, service_account_secret)

def download_video_from_youtube(link: str, path: str):
    yt = YouTube(link)
    video = yt.streams.get_highest_resolution()
    video.download(path)

    return yt.streams[0].title + ".mp4"


@app.on_event("startup")
def startup():
    app.models = models
    app.face_extractor = FaceExtractor(video_read_fn)
    app.input_size = 380
    app.strategy = confident_strategy
    app.s3 = s3

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Deepfake Backend Service",
        version="0.1.0",
        description="Inca test task",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


@app.get("/healthcheck")
async def healthcheck() -> bool:
    """Healthcheck."""
    return True


@router.post("/deepfake_predict")
async def deepface_predict(request: Request):
    """Predict if video has deepfake."""
    request = json.loads((await request.body()).decode())
    video_name = request["video_name"]
    url_type = request["url_type"]
    video_path = None
    if url_type == "uploaded":
        video_path = os.path.join(data_dir, video_name)
        app.s3.download_file(bucket_name, video_name, video_path)
    elif url_type == "youtube":
        youtube_video_name = download_video_from_youtube(video_name, data_dir)
        video_path = os.path.join(data_dir, youtube_video_name)

    y_pred, image = predict_on_video(face_extractor=app.face_extractor, video_path=video_path,
                              input_size=app.input_size,
                              batch_size=frames_per_video,
                              models=models, strategy=app.strategy, apply_compression=False)

    os.remove(video_path)
    with io.BytesIO() as buf:
        iio.imwrite(buf, image, plugin="pillow", format="JPEG")
        im_bytes = buf.getvalue()

    headers = {"is_deepfake": str(bool(int(np.round(y_pred))))}
    return Response(im_bytes, headers=headers, media_type='image/jpeg')

app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8080, reload=True)
