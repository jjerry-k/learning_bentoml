import os
from io import BytesIO

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

import torch

import bentoml
from bentoml import io

import utils

# After training, result directory is made in log directory
# example) log/{DATASET_NAME}/{YYYY_MM_DD}/{hh_mm_ss}

config_path = os.path.join('./log/flower_photos/2022_02_18/18_18_10', "config.yaml")
config = utils.config_parser(config_path)

classes = config["DATA"]["CLASSES"]

device = "cuda" if torch.cuda.is_available() else "cpu"

runner = bentoml.pytorch.load_runner(
                                    f'{config["DATA"]["NAME"]}:{config["COMMON"]["TAG"]}',
                                    name=config["DATA"]["NAME"],
                                    device_id=device
                                    )

svc = bentoml.Service(
                    name=f'{config["DATA"]["NAME"]}_service', # Bento Name
                    runners=[
                        runner,
                    ],
                )

@svc.api(input=io.Image(), output=io.JSON())
async def predict(input: PILImage) -> str:
    assert isinstance(input, PILImage)
    arr = np.array(input)/255.0
    arr = np.transpose(arr, [2, 0, 1]).astype("float32")

    predict = await runner.async_run(arr)
    predict = torch.softmax(predict, dim=0)
    predict = predict.numpy() if device == "cpu" else predict.cpu().numpy()
    
    predict_class_idx = predict.argmax()
    
    result_json = {
        "predicted_class": classes[predict_class_idx],
        "confidence": predict[predict_class_idx]}
    
    return result_json

@svc.api(input=io.File(), output=io.JSON())
async def predict4swagger(input):
    assert isinstance(input, bentoml._internal.types.FileLike)
    byte_image = BytesIO(input.bytes_)
    pil_image = Image.open(byte_image).convert("RGB")
    arr = np.array(pil_image)
    arr = np.array(arr)/255.0
    arr = np.transpose(arr, [2, 0, 1]).astype("float32")

    predict = await runner.async_run(arr)
    predict = torch.softmax(predict, dim=0)
    predict = predict.numpy() if device == "cpu" else predict.cpu().numpy()
    
    predict_class_idx = predict.argmax()
    
    result_json = {
        "predicted_class": classes[predict_class_idx],
        "confidence": predict[predict_class_idx]}
    
    return result_json