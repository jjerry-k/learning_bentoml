import os
import numpy as np
import PIL.Image
import torch
import bentoml

import utils

config_path = os.path.join('/home/coder/hdd/private/learning_bentoml/pytorch/log/flower_photos/2022_01_26/01_16_38', "config.yaml")
config = utils.config_parser(config_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Current device: {device}")
model = bentoml.pytorch.load(
    f'{config["DATA"]["NAME"]}:{config["COMMON"]["TAG"]}', 
    device_id="cuda:0")
print(f"model weight on gpu: {next(model.parameters()).is_cuda}")

# model = model.to(device)
# runner = bentoml.pytorch.load_runner(
#                                     f'{config["DATA"]["NAME"]}:{config["COMMON"]["TAG"]}',
#                                     name=config["DATA"]["NAME"],
#                                     device_id="cpu",
#                                     predict_fn_name="predict",
#                                     )


# img = PIL.Image.open("data/flower_photos/validation/rose/12240303_80d87f77a3_n.jpg")
# arr = np.array(img) / 255.0
# arr = arr.astype("float32")
# arr = np.transpose(arr, [2, 0, 1])
# arr = arr[np.newaxis]
# arr = torch.tensor(arr)
# arr = arr.to(device)
# # print(arr.shape)
# print(model.predict(arr))
# print(runner.run(arr))

