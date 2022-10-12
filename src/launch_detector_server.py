"""
launch a detector server
Usage:
    CUDA_VISIBLE_DEVICES=1 DETECTOR=configs/configs_detectors/cifar_atom.yml uvicorn launch_detector_server:app --port 8000

"""
import os

device_idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
pid = os.getpid()
print(f"pid: {pid}")
with open(f"server.{device_idx}.pid", "w") as f:
    f.write(f"{pid}")

import atexit


def exit_handler():
    print("Closing server")
    os.remove(f"server.{device_idx}.pid")


atexit.register(exit_handler)

import numpy as np
from fastapi import FastAPI
from omegaconf import OmegaConf
from pydantic import BaseModel

from attacks import get_detector


class InputData(BaseModel):
    batch_x: list


device = "cuda:0"  # it will be the only gpu inside this process.
detector_cfg_path = os.environ["DETECTOR"]
cfg = OmegaConf.load(detector_cfg_path)
print(detector_cfg_path)
model = get_detector(device=device, normalize=True, **cfg)

app = FastAPI()


@app.post("/")
async def root(item: InputData):
    # return {"message": "Hello World"}
    batch_x = torch.tensor(np.arraycitem.batch_x, device="cuda:0")
    pred = model.predict(batch_x)
    # return {'batch_x': batch_x.tolist(), 'pred': pred.tolist()}
    return {"pred": pred.tolist()}
