import argparse
import os
import typing as t

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

import torch
from torchvision import transforms

import bentoml
from bentoml.io import Image
from bentoml.io import NumpyNdarray

import utils

config_path = os.path.join('./log/flower_photos/2022_01_26/00_38_03', "config.yaml")
config = utils.config_parser(config_path)

device = "cuda" if torch.cuda.is_available() else "cpu"

runner = bentoml.pytorch.load_runner(
                                    f'{config["DATA"]["NAME"]}:{config["COMMON"]["TAG"]}',
                                    name=config["DATA"]["NAME"],
                                    device_id=device
                                    # predict_fn_name="predict",
                                    )

svc = bentoml.Service(
                    name=f'{config["DATA"]["NAME"]}_service',
                    runners=[
                        runner,
                    ],
                )

# # we will use cpu for prediction
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")

@svc.api(
input=NumpyNdarray(dtype="float32", enforce_dtype=True),
output=NumpyNdarray(dtype="int64"),
)
async def predict_ndarray(
    inp: "np.ndarray[t.Any, np.dtype[t.Any]]",
) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    # We are using greyscale image and our PyTorch model expect one
    # extra channel dimension
    # inp = np.expand_dims(inp, 0)
    arr = np.transpose(arr, [2, 0, 1]).astype("float32")/255.0
    output_tensor = await runner.async_run(arr)
    return output_tensor.numpy() if device == "cpu" else output_tensor.cpu().numpy()

@svc.api(input=Image(), output=NumpyNdarray(dtype="int64"))
async def predict_image(f: PILImage) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    assert isinstance(f, PILImage)
    arr = np.array(f)/255.0
    arr = np.transpose(arr, [2, 0, 1]).astype("float32")
    output_tensor = await runner.async_run(arr)
    return output_tensor.numpy() if device == "cpu" else output_tensor.cpu().numpy()