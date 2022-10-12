import json
from time import time

import numpy as np
import requests
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import ToTensor
from tqdm import tqdm

# Load the dataset
ds_cifar10 = CIFAR10(root="data", train=False, download=True, transform=ToTensor())
ds_cifar100 = CIFAR100(root="data", train=False, download=True, transform=ToTensor())

dl_cifar10 = DataLoader(ds_cifar10, batch_size=512, shuffle=False, num_workers=0)
dl_cifar100 = DataLoader(ds_cifar100, batch_size=128, shuffle=False)

to_tensor = ToTensor()

tt = 0
for i, (batch_x, target) in enumerate(tqdm(dl_cifar10)):
    batch_x = batch_x.permute(0, 2, 3, 1).numpy().tolist()
    data = {"batch_x": batch_x, "string": "a"}
    # data = {'batch_x': data.tolist(), 'string': 'hello world'}
    # Send data to server
    time_s = time()
    r = requests.post("http://localhost:8898", json=data)
    time_e = time()
    tt += time_e - time_s
    # r = requests.post(
    #         "http://localhost:8898",
    #         headers={"Content-Type": "application/json"},
    #         data=json.dumps(data))

    # Get the response
    if r.status_code == 200:
        pred = np.array(r.json()["pred"])
    else:
        print("Error: ", r.status_code)
        break

    if i == 10:
        break

print(pred.shape)
print(tt)
# batch_x = np.random.randn(100, 32, 32, 3).tolist()
# data = {'batch_x': batch_x, 'string': 'hello world'}
# print(a.json())
# print(type(a.json()['pred']))
