import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LogisticRegressionCV
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange


class MD(nn.Module):
    def __init__(self, net, net_type):
        super().__init__()
        self.net = net
        self.net_type = net_type
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, x, y, optimizer, clip_grad=None):
        self.train()

        optimizer.zero_grad()
        outputs = self.net(x)
        L = self.criterion(outputs, y)
        acc = 100 * (
            torch.sum((torch.argmax(outputs, dim=1) == y).float()) / outputs.shape[0]
        )
        L.backward()
        optimizer.step()

        return {"loss": L.item(), "train_acc_": acc.item()}

    def validation_step(self, x, y):
        self.eval()

        outputs = self.net(x)
        L = self.criterion(outputs, y)
        acc = 100 * (
            torch.sum((torch.argmax(outputs, dim=1) == y).float()) / outputs.shape[0]
        )
        return {"loss": L.item(), "val_acc_": acc.item()}

    def estimate_sample_mean_and_precisions(self, train_loader, device="cuda:0"):
        """
        compute sample mean and precision (inverse of covariance)
        return: sample_class_mean: list of class mean
                precision: list of precisions
        """
        model = self.net

        model.eval()
        temp_x = torch.rand(2, 3, 32, 32).to(device)
        temp_x = Variable(temp_x)
        temp_list = model.feature_list(temp_x)[1]
        num_output = len(temp_list)
        self.num_output = num_output
        feature_list = np.empty(num_output)
        count = 0
        for out in temp_list:
            feature_list[count] = out.size(1)
            count += 1

        import sklearn.covariance

        num_classes = model(temp_x).size(1)
        self.num_classes = num_classes

        model.eval()
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        correct, total = 0, 0
        num_output = len(feature_list)
        num_sample_per_class = np.empty(num_classes)
        num_sample_per_class.fill(0)
        list_features = []
        for i in range(num_output):
            temp_list = []
            for j in range(num_classes):
                temp_list.append(0)
            list_features.append(temp_list)

        for data, target in train_loader:
            total += data.size(0)
            data = data.to(device)
            with torch.no_grad():
                data = Variable(data)
                output, out_features = model.feature_list(data)

                # get hidden features
                for i in range(num_output):
                    out_features[i] = out_features[i].view(
                        out_features[i].size(0), out_features[i].size(1), -1
                    )
                    out_features[i] = torch.mean(out_features[i].data, 2)

                # compute the accuracy
                pred = output.data.max(1)[1]
                equal_flag = pred.eq(target.to(device)).cpu()
                correct += equal_flag.sum()

                # construct the sample matrix
                for i in range(data.size(0)):
                    label = target[i]
                    if num_sample_per_class[label] == 0:
                        out_count = 0
                        for out in out_features:
                            list_features[out_count][label] = out[i].view(1, -1)
                            out_count += 1
                    else:
                        out_count = 0
                        for out in out_features:
                            list_features[out_count][label] = torch.cat(
                                (list_features[out_count][label], out[i].view(1, -1)), 0
                            )
                            out_count += 1
                    num_sample_per_class[label] += 1

        sample_class_mean = []
        out_count = 0
        for num_feature in feature_list:
            temp_list = torch.Tensor(num_classes, int(num_feature)).to(device)
            for j in range(num_classes):
                temp_list[j] = torch.mean(list_features[out_count][j], 0)
            sample_class_mean.append(temp_list)
            out_count += 1

        precision = []
        for k in range(num_output):
            X = 0
            for i in range(num_classes):
                if i == 0:
                    X = list_features[k][i] - sample_class_mean[k][i]
                else:
                    X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

            # find inverse
            group_lasso.fit(X.cpu().numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float().to(device)
            precision.append(temp_precision)

        print("\n Training Accuracy:({:.2f}%)\n".format(100.0 * correct / total))

        self.sample_mean = sample_class_mean
        self.precision = precision

    def get_Mahalanobis_score(self, x, layer_index, magnitude):
        if not hasattr(self, "sample_mean"):
            print(
                "You should call estimate_sample_mean_and_precisions function first to compute mean and precisionts."
            )

        model = self.net
        model.eval()
        data = Variable(x, requires_grad=True)
        device = x.device

        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(self.num_classes):
            batch_sample_mean = self.sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean.to(device)
            term_gau = (
                -0.5
                * torch.mm(
                    torch.mm(zero_f.to(device), self.precision[layer_index].to(device)),
                    zero_f.to(device).t(),
                ).diag()
            )
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = (
            self.sample_mean[layer_index].to(device).index_select(0, sample_pred)
        )
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = (
            -0.5
            * torch.mm(
                torch.mm(
                    zero_f.to(device), Variable(self.precision[layer_index].to(device))
                ),
                zero_f.to(device).t(),
            ).diag()
        )
        loss = torch.mean(-pure_gau)
        loss.backward()

        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        if self.net_type == "densenet":
            gradient.index_copy_(
                1,
                torch.LongTensor([0]).to(device),
                gradient.index_select(1, torch.LongTensor([0]).to(device))
                / (63.0 / 255.0),
            )
            gradient.index_copy_(
                1,
                torch.LongTensor([1]).to(device),
                gradient.index_select(1, torch.LongTensor([1]).to(device))
                / (62.1 / 255.0),
            )
            gradient.index_copy_(
                1,
                torch.LongTensor([2]).to(device),
                gradient.index_select(1, torch.LongTensor([2]).to(device))
                / (66.7 / 255.0),
            )
        elif self.net_type == "resnet":
            gradient.index_copy_(
                1,
                torch.LongTensor([0]).to(device),
                gradient.index_select(1, torch.LongTensor([0]).to(device)) / (0.2023),
            )
            gradient.index_copy_(
                1,
                torch.LongTensor([1]).to(device),
                gradient.index_select(1, torch.LongTensor([1]).to(device)) / (0.1994),
            )
            gradient.index_copy_(
                1,
                torch.LongTensor([2]).to(device),
                gradient.index_select(1, torch.LongTensor([2]).to(device)) / (0.2010),
            )
        tempInputs = torch.add(data.data, -magnitude, gradient)

        with torch.no_grad():
            noise_out_features = model.intermediate_forward(
                Variable(tempInputs), layer_index
            )
            noise_out_features = noise_out_features.view(
                noise_out_features.size(0), noise_out_features.size(1), -1
            )
            noise_out_features = torch.mean(noise_out_features, 2)
            noise_gaussian_score = 0
            for i in range(self.num_classes):
                batch_sample_mean = self.sample_mean[layer_index][i]
                zero_f = noise_out_features.data - batch_sample_mean.to(device)
                term_gau = (
                    -0.5
                    * torch.mm(
                        torch.mm(
                            zero_f.to(device), self.precision[layer_index].to(device)
                        ),
                        zero_f.to(device).t(),
                    ).diag()
                )
                if i == 0:
                    noise_gaussian_score = term_gau.view(-1, 1)
                else:
                    noise_gaussian_score = torch.cat(
                        (noise_gaussian_score, term_gau.view(-1, 1)), 1
                    )

            noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)

        return noise_gaussian_score

    def predict_vector(self, x, magnitude=0.0):
        # m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005] hyper parameters for input processing
        # print('Noise: ' + str(magnitude))
        for i in range(self.num_output):
            M = self.get_Mahalanobis_score(x, i, magnitude)
            M = np.asarray(M.cpu(), dtype=np.float32)
            if i == 0:
                Mahalanobis = M.reshape((M.shape[0], -1))
            else:
                Mahalanobis = np.concatenate(
                    (Mahalanobis, M.reshape((M.shape[0], -1))), axis=1
                )
        Mahalanobis = np.asarray(Mahalanobis, dtype=np.float32)
        return -Mahalanobis

    def predict(self, x, layer_index=None, magnitude=0.0):
        device = x.device
        if (not hasattr(self, "lr")) or (layer_index is not None):
            if layer_index is None:
                print(
                    f"You shoul write layer index! 0 ~ 3 for densenet, and 0 ~ 4 for resnet!"
                )
            else:
                prob = (
                    torch.tensor(
                        self.predict_vector(x, magnitude=magnitude), dtype=torch.float32
                    )
                    .to(device)[:, layer_index]
                    .unsqueeze(1)
                )
        else:
            x = self.predict_vector(x, magnitude=magnitude)
            prob = self.lr.predict_proba(x)[:, 1]
        return torch.tensor(prob, dtype=torch.float32).to(device)

    def weight_tuning(
        self,
        in_loader,
        ood_loader,
        num_in=1000,
        num_out=100,
        magnitude=0.0,
        device="cuda:0",
    ):
        X = []
        Y = []
        for x, _ in in_loader:
            X.append(
                torch.tensor(
                    self.predict_vector(x.to(device), magnitude=magnitude),
                    dtype=torch.float32,
                )
                .clone()
                .detach()
            )
            Y.append(torch.tensor(0.0).repeat(x.size(0)))
        X = X[:num_in]
        Y = Y[:num_in]
        for x, _ in ood_loader:
            X.append(
                torch.tensor(
                    self.predict_vector(x.to(device), magnitude=magnitude),
                    dtype=torch.float32,
                )
                .clone()
                .detach()
            )
            Y.append(torch.tensor(1.0).repeat(x.size(0)))
        X = X[: num_in + num_out]
        Y = Y[: num_in + num_out]
        X = torch.cat(X, dim=0)
        Y = torch.cat(Y)
        randp = torch.randperm(len(X))
        X = X[randp][:1000]
        Y = Y[randp][:1000]
        X_train = X.view(X.size(0), -1)
        Y_train = Y
        lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
        self.lr = lr
