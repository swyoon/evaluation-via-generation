import json

import numpy as np
import requests

batch_x = np.array([[1, 2, 3], [4, 5.0, 6.0]]).tolist()
data = {"batch_x": batch_x, "string": "hello world"}
a = requests.post(
    "http://localhost:8898",
    headers={"Content-Type": "application/json"},
    data=json.dumps(data),
).text
print(a)
