import requests
import torch
import torch.nn as nn
from tqdm.auto import tqdm


class Detector(nn.Module):
    """A wrapper class for OOD detector.
    Main functions are:
        1. input pre-processing
        2. OOD score normalization"""

    def __init__(
        self,
        model,
        transform=None,
        mean=0,
        std=1,
        bound=-1,
        no_grad_predict=True,
        blackbox_only=False,
        up_bound=-1,
        use_rank=False,
    ):
        """
        no_grad_predict: `torch.no_grad()` when predict
        blackbox_only: only support `requires_grad=False`
        """
        super().__init__()
        self.model = model
        self.transform = transform
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        self.register_buffer("bound", torch.tensor(bound))
        self.no_grad_predict = no_grad_predict
        self.up_bound = up_bound
        self.use_rank = use_rank
        self.blackbox_only = blackbox_only

    def predict(self, x, normalize=True, binary_reward=None, requires_grad=False):
        if requires_grad and self.blackbox_only:
            raise ValueError("Blackbox detector does not support requires_grad=True")

        if requires_grad:
            return self._predict(x, normalize=normalize, binary_reward=binary_reward)

        if self.no_grad_predict:
            with torch.no_grad():
                return self._predict(
                    x, normalize=normalize, binary_reward=binary_reward
                )
        else:
            return self._predict(x, normalize=normalize, binary_reward=binary_reward)

    def _predict(self, x, normalize=True, binary_reward=None):
        if self.transform is not None:
            x = self.transform(x)
        score = self.model.predict(x)
        if normalize:
            score = self.normalize(score)
        if binary_reward is not None:
            index = score > binary_reward
            score[index] = 1
            score[~index] = -1
        return score

    def normalize(self, score):
        if self.use_rank:
            return torch.bucketize(score, self.in_score) / len(self.in_score) * 2 - 1

        else:
            normed = (score - self.mean) / self.std
            if self.bound > 0:
                normed.clip_(-self.bound, self.bound)
            if self.up_bound > 0:
                normed.clip_(max=self.up_bound)
            return normed

    # def learn_normalization(self, dataloader=None, device=None, use_grad=False):
    #     """compute normalization parameters for detector score"""
    #     l_score = []
    #     for xx, _ in tqdm(dataloader):
    #         if device is not None:
    #             xx = xx.to(device)
    #         if use_grad:
    #             l_score.append(self.predict(xx, normalize=False).detach())
    #         else:
    #             try:
    #                 with torch.no_grad():
    #                     l_score.append(self.predict(xx, normalize=False).detach())
    #             except:
    #                 l_score.append(self.predict(xx, normalize=False).detach())
    #     score = torch.cat(l_score)
    #     if self.use_rank:
    #         self.in_score = torch.sort(score).values
    #     else:
    #         mean_score = torch.mean(score)
    #         std_score = torch.std(score) + 1e-3
    #         self.mean = mean_score
    #         self.std = std_score

    #     normed_score = self.normalize(score)
    #     return normed_score

    # def save_normalization(self, norm_path):
    #     if self.use_rank:
    #         torch.save(self.in_score, norm_path)
    #         print("save rank normalization info")
    #     else:
    #         torch.save([self.mean, self.std], norm_path)
    #         print("save std normalization info")

    # def load_normalization(self, norm_path, device):
    #     if self.use_rank:
    #         self.in_score = torch.load(norm_path, map_location=device)
    #         print("load rank normalization info")
    #     else:
    #         self.mean, self.std = torch.load(norm_path, map_location=device)
    #         print("load std normalization info")

    def load_normalization(self, norm_path, device):
        in_score = torch.load(norm_path)
        mean_score = torch.mean(in_score)
        std_score = torch.std(in_score) + 1e-3
        self.mean = mean_score.to(device)
        self.std = std_score.to(device)


class DetectorClient:
    """A wrapper of a detector that communicate with a server"""

    def __init__(self, host, port):
        self.host = host
        self.port = port

    def predict(self, x):
        """
        x: torch.Tensor [batch, C, H, W]
        """
        data = {"batch_x": x.numpy().tolist()}
        r = requests.post(f"http://{self.host}:{self.port}/predict", json=data)
        if r.status_code == 200:
            pred = torch.tensor(r.json()["pred"], device=x.device)
            assert len(pred) == len(x)
            return pred
        else:
            raise RuntimeError(f"DetectorClient: {r.status_code} {r.text}")


class EnsembleDetector(nn.Module):
    """A wrapper class for Ensemble model of OOD detectors.
    Main functions are:
        1. input pre-processing
        2. OOD score normalization"""

    def __init__(
        self,
        l_model,
        mean=0,
        std=1,
        bound=-1,
        up_bound=-1,
        use_rank=False,
        agg="max",
        no_grad=False,
    ):
        """
        no_grad: `torch.no_grad()` when predict
        agg: aggregation method
        """
        super().__init__()
        self.l_model = l_model
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        self.register_buffer("bound", torch.tensor(bound))
        self.up_bound = up_bound
        self.use_rank = use_rank
        self.agg = agg
        self.no_grad = no_grad

    def predict(self, x, normalize=True, binary_reward=None):
        l_predict = []
        for model in self.l_model:
            if model.no_grad:
                f = torch.no_grad()(model._predict)
            else:
                f = model._predict
            l_predict.append(f(x, normalize=True, binary_reward=binary_reward))
        pred = torch.stack(l_predict)
        pred = self.aggregate(pred)
        if normalize:
            return self.normalize(pred)
        else:
            return pred

    def aggregate(self, pred):
        """pred: (n_model) x (n_example)"""
        if self.agg == "mean":
            return pred.mean(dim=0)
        elif self.agg == "max":
            return pred.max(dim=0).values
        else:
            raise ValueError(f"Invalid aggregation {self.agg}")

    def normalize(self, score):
        if self.use_rank:
            return torch.bucketize(score, self.in_score) / len(self.in_score) * 2 - 1

        else:
            normed = (score - self.mean) / self.std
            if self.bound > 0:
                normed.clip_(-self.bound, self.bound)
            if self.up_bound > 0:
                normed.clip_(max=self.up_bound)
            return normed

    def learn_normalization(self, dataloader=None, device=None, use_grad=False):
        """compute normalization parameters for detector score"""
        l_score = []
        for xx, _ in tqdm(dataloader):
            if device is not None:
                xx = xx.to(device)
            l_score.append(self.predict(xx, normalize=False).detach())

        score = torch.cat(l_score)
        if self.use_rank:
            self.in_score = torch.sort(score).values
        else:
            mean_score = torch.mean(score)
            std_score = torch.std(score) + 1e-3
            self.mean = mean_score
            self.std = std_score

        normed_score = self.normalize(score)
        return score
