from typing import Union

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


class InputData(BaseModel):
    string: str
    batch_x: list


app = FastAPI()


@app.post("/")
async def root(item: InputData):
    # return {"message": "Hello World"}
    batch_x = np.array(item.batch_x)
    return {"batch_x": batch_x.tolist()}
