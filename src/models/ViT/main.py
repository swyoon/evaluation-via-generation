from typing import Union

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from vit_ood import get_model


class InputData(BaseModel):
    string: str
    batch_x: list


app = FastAPI()

model = get_model()


@app.post("/")
async def root(item: InputData):
    # return {"message": "Hello World"}
    batch_x = np.array(item.batch_x)
    pred = model.predict(batch_x)
    # return {'batch_x': batch_x.tolist(), 'pred': pred.tolist()}
    return {"pred": pred.tolist()}
