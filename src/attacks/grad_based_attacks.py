import numpy as np
import torch
from tqdm.auto import tqdm

"""eval tools"""


def attack(m, method, x, y=None, device=None, use_label=False):
    eps = 0.01
    iterations = 200
    stepsize = 0.1
    budget = 5
    restarts = 5

    ### YHLEE ###
    class Loss_for_attack(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x, y, x1, y1, reduction="mean"):
            return self.model.predict(x, requires_grad=True)

    loss = Loss_for_attack(m)
    #############

    noise = DeContraster(eps)
    # attack1 = modified_MonotonePGD(m, eps, iterations, stepsize, momentum=0.9,
    #             norm='inf', normalize_grad=False, restarts=0,
    #             init_noise_generator=noise, save_trajectory=False, use_label=use_label)
    attack1 = MonotonePGD(
        eps,
        iterations,
        stepsize,
        10,
        momentum=0.9,
        norm="inf",
        loss=loss,
        normalize_grad=False,
        early_stopping=0,
        restarts=0,
        init_noise_generator=noise,
        model=None,
        save_trajectory=False,
    )

    noise = UniformNoiseGenerator(min=-eps, max=eps)
    # attack2 = modified_MonotonePGD(m, eps, iterations, stepsize, momentum=0.9,
    #             norm='inf', normalize_grad=False, restarts=0,
    #             init_noise_generator=noise, save_trajectory=False, use_label=use_label)
    attack2 = MonotonePGD(
        eps,
        iterations,
        stepsize,
        10,
        momentum=0.9,
        norm="inf",
        loss=loss,
        normalize_grad=False,
        early_stopping=0,
        restarts=0,
        init_noise_generator=noise,
        model=None,
        save_trajectory=False,
    )

    noise = NormalNoiseGenerator(sigma=1e-4)
    # attack3 = modified_MonotonePGD(m, eps, iterations, stepsize, momentum=0.9,
    #             norm='inf', normalize_grad=False, restarts=0,
    #             init_noise_generator=noise, save_trajectory=False, use_label=use_label)
    attack3 = MonotonePGD(
        eps,
        iterations,
        stepsize,
        10,
        momentum=0.9,
        norm="inf",
        loss=loss,
        normalize_grad=False,
        early_stopping=0,
        restarts=0,
        init_noise_generator=noise,
        model=None,
        save_trajectory=False,
    )

    # attack = APGDAttack(
    #     m, n_iter=100*budget, n_iter_2=22*budget, n_iter_min=6*budget,
    #     size_decr=3, norm='Linf', eps=eps, seed=0, eot_iter=1, thr_decr=.75,
    #     check_impr=False, device=device)

    attack = APGDAttack(
        None,
        n_iter=100 * budget,
        n_iter_2=22 * budget,
        n_iter_min=6 * budget,
        size_decr=3,
        norm="Linf",
        n_restarts=restarts,
        eps=eps,
        show_loss=False,
        seed=0,
        loss=loss,
        show_acc=False,
        eot_iter=1,
        save_steps=False,
        save_dir="./results/",
        thr_decr=0.75,
        check_impr=False,
        normalize_logits=False,
        device=device,
        apply_softmax=None,
        classes=10,
    )

    list_pred = []
    for att in [attack1, attack2, attack3]:
        x_adv = att.perturb(x, y=y)
        pred = method(x_adv)
        list_pred.append(pred.unsqueeze(0))

    x_adv = attack.perturb(x, y=y)
    pred = method(x_adv)
    list_pred.append(pred.unsqueeze(0))

    best_attack_pred = torch.min(torch.cat(list_pred, dim=0), 0).values
    return best_attack_pred


def batch_attack(
    m,
    dl,
    device,
    flatten=False,
    method="predict",
    input_type="first",
    no_grad=False,
    use_label=False,
    **kwargs,
):
    method = getattr(m, method)
    l_result = []
    for batch in tqdm(dl):
        if input_type == "first":
            x = batch[0]
            y = batch[1]
        if no_grad:
            with torch.no_grad():
                if flatten:
                    x = x.view(len(x), -1)
                pred = (
                    attack(
                        m,
                        method,
                        x.cuda(device),
                        y=y.cuda(device),
                        device=device,
                        use_label=use_label,
                    )
                    .detach()
                    .cpu()
                )
        else:
            if flatten:
                x = x.view(len(x), -1)
            pred = (
                attack(
                    m,
                    method,
                    x.cuda(device),
                    y=y.cuda(device),
                    device=device,
                    use_label=use_label,
                )
                .detach()
                .cpu()
            )

        l_result.append(pred)
    return torch.cat(l_result)


"""PGD attacks tools"""


def normalize_perturbation(perturbation, p):
    if p == "inf":
        return perturbation.sign()
    elif p == 2 or p == 2.0:
        bs = perturbation.shape[0]
        pert_flat = perturbation.view(bs, -1)
        pert_normalized = torch.nn.functional.normalize(pert_flat, p=p, dim=1)
        return pert_normalized.view_as(perturbation)
    else:
        raise NotImplementedError("Projection only supports l2 and inf norm")


def project_perturbation(perturbation, eps, p):
    if p == "inf":
        mask = perturbation.abs() > eps
        pert_normalized = perturbation
        pert_normalized[mask] = eps * perturbation[mask].sign()
        return pert_normalized
    elif p == 2 or p == 2.0:
        # TODO use torch.renorm
        bs = perturbation.shape[0]
        pert_flat = perturbation.view(bs, -1)
        norm = torch.norm(perturbation.view(bs, -1), dim=1) + 1e-10
        mask = norm > eps
        pert_normalized = pert_flat
        pert_normalized[mask, :] = (eps / norm[mask, None]) * pert_flat[mask, :]
        return pert_normalized.view_as(perturbation)
    else:
        raise NotImplementedError("Projection only supports l2 and inf norm")


def calculate_smart_lr(prev_mean_lr, lr_accepted, lr_decay, iterations, max_lr):
    accepted_idcs = lr_accepted > 0
    if torch.sum(accepted_idcs).item() > 0:
        new_lr = 0.5 * (prev_mean_lr + torch.mean(lr_accepted[lr_accepted > 0]).item())
    else:
        new_lr = prev_mean_lr * (lr_decay**iterations)

    new_lr = min(max_lr, new_lr)
    return new_lr


def logits_diff_loss(out, y_oh, reduction="mean"):
    # out: model output
    # y_oh: targets in one hot encoding
    # confidence:
    out_real = torch.sum((out * y_oh), 1)
    out_other = torch.max(out * (1.0 - y_oh) - y_oh * 100000000.0, 1)[0]

    diff = out_other - out_real

    return TrainLoss.reduce(diff, reduction)


def create_early_stopping_mask(out, y, conf_threshold, targeted):
    finished = False
    conf, pred = torch.max(torch.nn.functional.softmax(out, dim=1), 1)
    conf_mask = conf > conf_threshold
    if targeted:
        correct_mask = torch.eq(y, pred)
    else:
        correct_mask = ~torch.eq(y, pred)

    mask = 1.0 - (conf_mask & correct_mask).float()

    if sum(1.0 - mask) == out.shape[0]:
        finished = True

    mask = mask[(...,) + (None,) * 3]
    return finished, mask


class Adversarial_attack:
    def __init__(self, loss, num_classes, model=None, save_trajectory=False):
        # loss should either be a string specifying one of the predefined loss functions
        # OR
        # a custom loss function taking 4 arguments as train_loss class
        self.loss = loss
        self.save_trajectory = False
        self.last_trajectory = None
        self.num_classes = num_classes
        if model is not None:
            self.model = model
        else:
            self.model = None

    def __call__(self, *args, **kwargs):
        return self.perturb(*args, **kwargs)

    def set_loss(self, loss):
        self.loss = loss

    def _get_loss_f(self, x, y, targeted, reduction):
        # x, y original data / target
        # targeted whether to use a targeted attack or not
        # reduction: reduction to use: 'sum', 'mean', 'none'
        if isinstance(self.loss, str):
            if self.loss.lower() == "crossentropy":
                if not targeted:
                    l_f = lambda data, data_out: -torch.nn.functional.cross_entropy(
                        data_out, y, reduction=reduction
                    )
                else:
                    l_f = lambda data, data_out: torch.nn.functional.cross_entropy(
                        data_out, y, reduction=reduction
                    )
            elif self.loss.lower() == "logitsdiff":
                if not targeted:
                    y_oh = torch.nn.functional.one_hot(y, self.num_classes)
                    y_oh = y_oh.float()
                    l_f = lambda data, data_out: -logits_diff_loss(
                        data_out, y_oh, reduction=reduction
                    )
                else:
                    y_oh = torch.nn.functional.one_hot(y, self.num_classes)
                    y_oh = y_oh.float()
                    l_f = lambda data, data_out: logits_diff_loss(
                        data_out, y_oh, reduction=reduction
                    )
            else:
                raise ValueError(f"Loss {self.loss} not supported")
        else:
            # for monotone pgd, this has to be per patch example, not mean
            l_f = lambda data, data_out: self.loss(
                data, data_out, x, y, reduction=reduction
            )

        return l_f

    def get_config_dict(self):
        raise NotImplementedError()

    def get_last_trajectory(self):
        if not self.save_trajectory or self.last_trajectory is None:
            raise AssertionError()
        else:
            return self.last_trajectory

    def __get_trajectory_depth(self):
        raise NotImplementedError()

    def set_model(self, model):
        self.model = model

    def check_model(self):
        if self.model is None:
            raise RuntimeError("Attack model not set")

    def perturb(self, x, y, targeted=False):
        # force child class implementation
        raise NotImplementedError()


class Restart_attack(Adversarial_attack):
    # Base class for attacks that start from different initial values
    # Make sure that they MINIMIZE the given loss function
    def __init__(self, loss, restarts, num_classes, model=None, save_trajectory=False):
        super().__init__(
            loss, num_classes, model=model, save_trajectory=save_trajectory
        )
        self.restarts = restarts

    def perturb_inner(self, x, y, targeted=False):
        # force child class implementation
        raise NotImplementedError()

    def perturb(self, x, y, targeted=False):
        # base class method that handles various restarts

        ### YHLEE ###
        # self.check_model()

        # is_train = self.model.training
        # is_train = False
        # self.model.eval()
        # for module in self.model.modules():
        #     if isinstance(module, torch.nn.BatchNorm2d):
        #         module.track_running_stats = False
        #############

        restarts_data = x.new_empty((1 + self.restarts,) + x.shape)
        restarts_objs = x.new_empty((1 + self.restarts, x.shape[0]))

        if self.save_trajectory:
            self.last_trajectory = None
            trajectories_shape = (
                (self.restarts,) + (self.__get_trajectory_depth(),) + x.shape
            )
            restart_trajectories = x.new_empty(trajectories_shape)

        for k in range(1 + self.restarts):
            k_data, k_obj, k_trajectory = self.perturb_inner(x, y, targeted=targeted)
            restarts_data[k, :] = k_data
            restarts_objs[k, :] = k_obj
            if self.save_trajectory:
                restart_trajectories[k, :] = k_trajectory

        bs = x.shape[0]
        best_idx = torch.argmin(restarts_objs, 0)
        best_data = restarts_data[best_idx, range(bs), :]

        if self.save_trajectory:
            self.last_trajectory = restart_trajectories[best_idx, :, range(bs), :]

        ### YHLEE ###
        # #reset model status
        # if is_train:
        #     self.model.train()
        # else:
        #     self.model.eval()
        # for module in self.model.modules():
        #     if isinstance(module, torch.nn.BatchNorm2d):
        #         module.track_running_stats = True
        #############

        return best_data


class MonotonePGD(Restart_attack):
    def __init__(
        self,
        eps,
        iterations,
        stepsize,
        num_classes,
        momentum=0.9,
        lr_smart=False,
        lr_decay=0.5,
        lr_gain=1.1,
        norm="inf",
        loss="CrossEntropy",
        normalize_grad=False,
        early_stopping=0,
        restarts=0,
        init_noise_generator=None,
        model=None,
        save_trajectory=False,
    ):
        super().__init__(
            loss, restarts, num_classes, model=model, save_trajectory=save_trajectory
        )
        # loss either pass 'CrossEntropy' or 'LogitsDiff' or custom loss function
        self.eps = eps
        self.iterations = iterations
        self.stepsize = stepsize
        self.momentum = momentum
        self.norm = norm
        self.normalize_grad = normalize_grad
        self.early_stopping = early_stopping
        self.init_noise_generator = init_noise_generator
        self.lr_smart = lr_smart
        self.prev_mean_lr = stepsize
        self.lr_decay = lr_decay  # stepsize decay
        self.lr_gain = lr_gain

    def __get_trajectory_depth(self):
        return self.iterations + 1

    def get_config_dict(self):
        dict = {}
        dict["type"] = "PGD"
        dict["eps"] = self.eps
        dict["iterations"] = self.iterations
        dict["stepsize"] = self.stepsize
        dict["norm"] = self.norm
        if isinstance(self.loss, str):
            dict["loss"] = self.loss
        dict["restarts"] = self.restarts
        # dict['init_sigma'] = self.init_sigma
        dict["lr_gain"] = self.lr_gain
        dict["lr_decay"] = self.lr_decay
        return dict

    def perturb_inner(self, x, y, targeted=False):
        l_f = self._get_loss_f(x, y, targeted, "none")

        if self.lr_smart:
            lr_accepted = -1 * x.new_ones(self.iterations, x.shape[0])
            lr = self.prev_mean_lr * x.new_ones(x.shape[0])
        else:
            lr = self.stepsize * x.new_ones(x.shape[0])

        # initialize perturbation
        if self.init_noise_generator is None:
            pert = torch.zeros_like(x)
        else:
            pert = self.init_noise_generator(x)
            pert = torch.clamp(x + pert, 0, 1) - x  # box constraint
            pert = project_perturbation(pert, self.eps, self.norm)

        # TODO fix the datatype here !!!
        prev_loss = 1e13 * x.new_ones(x.shape[0], dtype=torch.float)

        prev_pert = pert.clone().detach()
        prev_velocity = torch.zeros_like(pert)
        velocity = torch.zeros_like(pert)

        # trajectory container
        if self.save_trajectory:
            trajectory = torch.zeros((self.iterations + 1,) + x.shape, device=x.device)
            trajectory[0, :] = x
        else:
            trajectory = None

        for i in range(self.iterations + 1):
            pert.requires_grad_(True)
            with torch.enable_grad():
                data = x + pert
                # out = self.model(data)
                loss_expanded = l_f(data, None)
                loss = torch.mean(loss_expanded)
                grad = torch.autograd.grad(loss, pert)[0]

            with torch.no_grad():
                loss_increase_idx = loss_expanded > prev_loss

                pert[loss_increase_idx, :] = (
                    prev_pert[loss_increase_idx, :].clone().detach()
                )
                loss_expanded[loss_increase_idx] = (
                    prev_loss[loss_increase_idx].clone().detach()
                )
                prev_pert = pert.clone().detach()
                prev_loss = loss_expanded
                # previous velocity always holds the last accepted velocity vector
                # velocity the one used for the last update that might have been rejected
                velocity[loss_increase_idx, :] = prev_velocity[loss_increase_idx, :]
                prev_velocity = velocity.clone().detach()

                if i > 0:
                    # use standard lr in firt iteration
                    lr[loss_increase_idx] *= self.lr_decay
                    lr[~loss_increase_idx] *= self.lr_gain

                if i == self.iterations:
                    break

                if self.lr_smart:
                    lr_accepted[i, ~loss_increase_idx] = lr[~loss_increase_idx]

                if self.early_stopping > 0:
                    finished, mask = create_early_stopping_mask(
                        out, y, self.early_stopping, targeted
                    )
                    if finished:
                        break
                else:
                    mask = 1.0

                # pgd on given loss
                if self.normalize_grad:
                    if self.momentum > 0:
                        # https://arxiv.org/pdf/1710.06081.pdf the l1 normalization follows the momentum iterative method
                        l1_norm_gradient = 1e-10 + torch.sum(
                            grad.abs().view(x.shape[0], -1), dim=1
                        ).view(-1, 1, 1, 1)
                        velocity = self.momentum * velocity + grad / l1_norm_gradient
                    else:
                        velocity = grad
                    norm_velocity = normalize_perturbation(velocity, self.norm)
                else:
                    # velocity update as in pytorch https://pytorch.org/docs/stable/optim.html
                    velocity = self.momentum * velocity + grad
                    norm_velocity = velocity

                pert = pert - mask * lr[:, None, None, None] * norm_velocity
                pert = project_perturbation(pert, self.eps, self.norm)
                pert = torch.clamp(x + pert, 0, 1) - x  # box constraint

                if self.save_trajectory:
                    trajectory[i + 1] = x + pert

        if self.lr_smart:
            self.prev_mean_lr = calculate_smart_lr(
                self.prev_mean_lr,
                lr_accepted,
                self.lr_decay,
                self.iterations,
                2 * self.eps,
            )

        return data, loss_expanded, trajectory


# class modified_MonotonePGD():
#     def __init__(self, model, eps, iterations, stepsize, momentum=0.9,
#                  lr_smart=False, lr_decay=0.5, lr_gain=1.1,
#                  norm='inf', normalize_grad=False, restarts=0,
#                  init_noise_generator=None, save_trajectory=False, use_label=False):
#         super().__init__()
#         #loss either pass 'CrossEntropy' or 'LogitsDiff' or custom loss function
#         self.eps = eps
#         self.iterations = iterations
#         self.stepsize = stepsize
#         self.momentum = momentum
#         self.norm = norm
#         self.normalize_grad = normalize_grad
#         self.init_noise_generator = init_noise_generator
#         self.lr_smart = lr_smart
#         self.prev_mean_lr = stepsize
#         self.lr_decay = lr_decay #stepsize decay
#         self.lr_gain = lr_gain
#         self.restarts = restarts
#         self.save_trajectory = False
#         self.last_trajectory = None
#         self.model = model
#         self.use_label = use_label
#         if self.use_label:
#             self.loss = 'CrossEntropy'
#             self.num_classes = 10
#             self.early_stopping = 0

#     def _get_loss_f(self, x, y, targeted, reduction):
#         #x, y original data / target
#         #targeted whether to use a targeted attack or not
#         #reduction: reduction to use: 'sum', 'mean', 'none'
#         if isinstance(self.loss, str):
#             if self.loss.lower() =='crossentropy':
#                 if not targeted:
#                     l_f = lambda data, data_out: -torch.nn.functional.cross_entropy(data_out, y, reduction=reduction)
#                 else:
#                     l_f = lambda data, data_out: torch.nn.functional.cross_entropy(data_out, y, reduction=reduction)
#             elif self.loss.lower() == 'logitsdiff':
#                 if not targeted:
#                     y_oh = torch.nn.functional.one_hot(y, self.num_classes)
#                     y_oh = y_oh.float()
#                     l_f = lambda data, data_out: -logits_diff_loss(data_out, y_oh, reduction=reduction)
#                 else:
#                     y_oh = torch.nn.functional.one_hot(y, self.num_classes)
#                     y_oh = y_oh.float()
#                     l_f = lambda data, data_out: logits_diff_loss(data_out, y_oh, reduction=reduction)
#             else:
#                 raise ValueError(f'Loss {self.loss} not supported')
#         else:
#             #for monotone pgd, this has to be per patch example, not mean
#             l_f = lambda data, data_out: self.loss(data, data_out, x, y, reduction=reduction)
#         return l_f

#     def check_model(self):
#         if self.model is None:
#             raise RuntimeError('Attack model not set')

#     def __get_trajectory_depth(self):
#         return self.iterations + 1

#     def perturb_inner(self, x, y=None, targeted=False):
#         if self.use_label:
#             l_f = self._get_loss_f(x, y, targeted, 'none')

#             if self.lr_smart:
#                 lr_accepted = -1 * x.new_ones(self.iterations, x.shape[0])
#                 lr = self.prev_mean_lr * x.new_ones(x.shape[0])
#             else:
#                 lr = self.stepsize * x.new_ones(x.shape[0])

#             #initialize perturbation
#             if self.init_noise_generator is None:
#                 pert = torch.zeros_like(x)
#             else:
#                 pert = self.init_noise_generator(x)
#                 pert = torch.clamp(x + pert, 0, 1) - x  # box constraint
#                 pert = project_perturbation(pert, self.eps, self.norm)

#             #TODO fix the datatype here !!!
#             prev_loss = 1e13 * x.new_ones(x.shape[0], dtype=torch.float)

#             prev_pert = pert.clone().detach()
#             prev_velocity = torch.zeros_like(pert)
#             velocity = torch.zeros_like(pert)

#             #trajectory container
#             if self.save_trajectory:
#                 trajectory = torch.zeros((self.iterations + 1,) + x.shape, device=x.device)
#                 trajectory[0, :] = x
#             else:
#                 trajectory = None

#             for i in range(self.iterations + 1):
#                 pert.requires_grad_(True)
#                 with torch.enable_grad():
#                     data = x + pert
#                     out = self.model.model.network(data)

#                     loss_expanded = l_f(data, out)
#                     loss = torch.mean(loss_expanded)
#                     grad = torch.autograd.grad(loss, pert)[0]

#                 with torch.no_grad():
#                     loss_increase_idx = loss_expanded > prev_loss

#                     pert[loss_increase_idx, :] = prev_pert[loss_increase_idx, :].clone().detach()
#                     loss_expanded[loss_increase_idx] = prev_loss[loss_increase_idx].clone().detach()
#                     prev_pert = pert.clone().detach()
#                     prev_loss = loss_expanded
#                     #previous velocity always holds the last accepted velocity vector
#                     #velocity the one used for the last update that might have been rejected
#                     velocity[loss_increase_idx, :] = prev_velocity[loss_increase_idx, :]
#                     prev_velocity = velocity.clone().detach()

#                     if i > 0:
#                         #use standard lr in firt iteration
#                         lr[loss_increase_idx] *= self.lr_decay
#                         lr[~loss_increase_idx] *= self.lr_gain

#                     if i == self.iterations:
#                         break

#                     if self.lr_smart:
#                         lr_accepted[i, ~loss_increase_idx] = lr[~loss_increase_idx]

#                     if self.early_stopping > 0:
#                         finished, mask = create_early_stopping_mask(out, y, self.early_stopping, targeted)
#                         if finished:
#                             break
#                     else:
#                         mask = 1.

#                     #pgd on given loss
#                     if self.normalize_grad:
#                         if self.momentum > 0:
#                             #https://arxiv.org/pdf/1710.06081.pdf the l1 normalization follows the momentum iterative method
#                             l1_norm_gradient = 1e-10 + torch.sum(grad.abs().view(x.shape[0], -1), dim=1).view(-1,1,1,1)
#                             velocity = self.momentum * velocity + grad / l1_norm_gradient
#                         else:
#                             velocity = grad
#                         norm_velocity = normalize_perturbation(velocity, self.norm)
#                     else:
#                         # velocity update as in pytorch https://pytorch.org/docs/stable/optim.html
#                         velocity = self.momentum * velocity + grad
#                         norm_velocity = velocity

#                     pert = pert - mask * lr[:,None,None,None] * norm_velocity
#                     pert = project_perturbation(pert, self.eps, self.norm)
#                     pert = torch.clamp(x + pert, 0, 1) - x #box constraint

#                     if self.save_trajectory:
#                         trajectory[i + 1] = x + pert

#             if self.lr_smart:
#                 self.prev_mean_lr = calculate_smart_lr(self.prev_mean_lr, lr_accepted, self.lr_decay, self.iterations, 2 * self.eps)

#             return data, loss_expanded, trajectory
#         else:
#             l_f = self.model.predict(x)

#             if self.lr_smart:
#                 lr_accepted = -1 * x.new_ones(self.iterations, x.shape[0])
#                 lr = self.prev_mean_lr * x.new_ones(x.shape[0])
#             else:
#                 lr = self.stepsize * x.new_ones(x.shape[0])

#             #initialize perturbation
#             if self.init_noise_generator is None:
#                 pert = torch.zeros_like(x)
#             else:
#                 pert = self.init_noise_generator(x)
#                 pert = torch.clamp(x + pert, 0, 1) - x  # box constraint
#                 pert = project_perturbation(pert, self.eps, self.norm)

#             #TODO fix the datatype here !!!
#             prev_loss = 1e13 * x.new_ones(x.shape[0], dtype=torch.float)

#             prev_pert = pert.clone().detach()
#             prev_velocity = torch.zeros_like(pert)
#             velocity = torch.zeros_like(pert)

#             #trajectory container
#             if self.save_trajectory:
#                 trajectory = torch.zeros((self.iterations + 1,) + x.shape, device=x.device)
#                 trajectory[0, :] = x
#             else:
#                 trajectory = None

#             for i in range(self.iterations + 1):
#                 pert.requires_grad_(True)
#                 with torch.enable_grad():
#                     data = x + pert
#                     loss_expanded = self.model.predict(data)
#                     loss = torch.mean(loss_expanded)
#                     grad = torch.autograd.grad(loss, pert)[0]

#                 with torch.no_grad():
#                     loss_increase_idx = loss_expanded > prev_loss

#                     pert[loss_increase_idx, :] = prev_pert[loss_increase_idx, :].clone().detach()
#                     loss_expanded[loss_increase_idx] = prev_loss[loss_increase_idx].clone().detach()
#                     prev_pert = pert.clone().detach()
#                     prev_loss = loss_expanded
#                     #previous velocity always holds the last accepted velocity vector
#                     #velocity the one used for the last update that might have been rejected
#                     velocity[loss_increase_idx, :] = prev_velocity[loss_increase_idx, :]
#                     prev_velocity = velocity.clone().detach()

#                     if i > 0:
#                         #use standard lr in firt iteration
#                         lr[loss_increase_idx] *= self.lr_decay
#                         lr[~loss_increase_idx] *= self.lr_gain

#                     if i == self.iterations:
#                         break

#                     if self.lr_smart:
#                         lr_accepted[i, ~loss_increase_idx] = lr[~loss_increase_idx]

#                     #pgd on given loss
#                     if self.normalize_grad:
#                         if self.momentum > 0:
#                             #https://arxiv.org/pdf/1710.06081.pdf the l1 normalization follows the momentum iterative method
#                             l1_norm_gradient = 1e-10 + torch.sum(grad.abs().view(x.shape[0], -1), dim=1).view(-1,1,1,1)
#                             velocity = self.momentum * velocity + grad / l1_norm_gradient
#                         else:
#                             velocity = grad
#                         norm_velocity = normalize_perturbation(velocity, self.norm)
#                     else:
#                         # velocity update as in pytorch https://pytorch.org/docs/stable/optim.html
#                         velocity = self.momentum * velocity + grad
#                         norm_velocity = velocity

#                     pert = pert - lr[:,None,None,None] * norm_velocity
#                     pert = project_perturbation(pert, self.eps, self.norm)
#                     pert = torch.clamp(x + pert, 0, 1) - x #box constraint

#                     if self.save_trajectory:
#                         trajectory[i + 1] = x + pert

#             if self.lr_smart:
#                 self.prev_mean_lr = calculate_smart_lr(self.prev_mean_lr, lr_accepted, self.lr_decay, self.iterations, 2 * self.eps)

#             return data, loss_expanded, trajectory

#     def perturb(self, x, y=None, targeted=False):
#         #base class method that handles various restarts
#         self.check_model()
#         self.model.eval()

#         restarts_data = x.new_empty((1 + self.restarts,) + x.shape)
#         restarts_objs = x.new_empty((1 + self.restarts, x.shape[0]))

#         if self.save_trajectory:
#             self.last_trajectory = None
#             trajectories_shape = (self.restarts,) + (self.__get_trajectory_depth(),) + x.shape
#             restart_trajectories = x.new_empty(trajectories_shape)

#         for k in range(1 + self.restarts):
#             k_data, k_obj, k_trajectory = self.perturb_inner(x, y=y, targeted=targeted)
#             restarts_data[k, :] = k_data
#             restarts_objs[k, :] = k_obj
#             if self.save_trajectory:
#                 restart_trajectories[k, :] = k_trajectory

#         bs = x.shape[0]
#         best_idx = torch.argmin(restarts_objs, 0)
#         best_data = restarts_data[best_idx, range(bs), :]

#         if self.save_trajectory:
#             self.last_trajectory = restart_trajectories[best_idx, :, range(bs), :]

#         return best_data


class AdversarialNoiseGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, x):
        # generate noise matching the size of x
        raise NotImplementedError()


class DeContraster(AdversarialNoiseGenerator):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        diff = torch.clamp(
            x.mean(dim=(1, 2, 3))[:, None, None, None] - x, -self.eps, self.eps
        )
        return diff


class UniformNoiseGenerator(AdversarialNoiseGenerator):
    def __init__(self, min=0.0, max=1.0):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return (self.max - self.min) * torch.rand_like(x) + self.min


class NormalNoiseGenerator(AdversarialNoiseGenerator):
    def __init__(self, sigma=1.0, mu=0):
        super().__init__()
        self.sigma = sigma
        self.mu = mu

    def forward(self, x):
        return self.sigma * torch.randn_like(x) + self.mu


# '''APGD attacks tools'''
# class APGDAttack():
#     def __init__(self, model, n_iter=100, n_iter_2=22, n_iter_min=6, size_decr=3,
#                  norm='Linf', eps=0.3, show_loss=False, seed=0,
#                  eot_iter=1, thr_decr=.75, check_impr=False,
#                   device=torch.device('cuda:0')):
#         self.model = model
#         self.n_iter = n_iter
#         self.n_iter_2 = n_iter_2
#         self.n_iter_min = n_iter_min
#         self.size_decr = size_decr
#         self.eps = eps
#         self.norm = norm
#         self.show_loss = show_loss
#         self.verbose = True
#         self.seed = seed
#         self.eot_iter = eot_iter
#         self.thr_decr = thr_decr
#         self.check_impr = check_impr
#         self.device = device

#     def check_oscillation(self, x, j, k, y5, k3=0.5):
#         t = np.zeros(x.shape[1])
#         for counter5 in range(k):
#             t += x[j - counter5] > x[j - counter5 - 1]

#         return t <= k*k3*np.ones(t.shape), t > k*1.0*np.ones(t.shape)

#     def check_shape(self, x):
#         return x if len(x.shape) > 0 else np.expand_dims(x, 0)

#     def norm_to_interval(self, x):
#         return x / (x.max(dim=1, keepdim=True)[0] + 1e-12)


#     def custom_loss(self, x, y=None):
#         x_sorted, ind_sorted = x.sort(dim=1)
#         ind = (ind_sorted[:, -1] == y).float()

#         return -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

#     def perturb(self, x_in):
#         x = x_in if len(x_in.shape) == 4 else x_in.unsqueeze(0)

#         if self.norm == 'Linf':
#             t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
#             x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
#         elif self.norm == 'L2':
#             t = torch.randn(x.shape).to(self.device).detach()
#             x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e12)
#         x_adv = x_adv.clamp(0., 1.)
#         x_best = x_adv.clone()
#         x_best_adv = x_adv.clone()
#         loss_steps = torch.zeros([self.n_iter, x.shape[0]])
#         loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]])
#         acc_steps = torch.zeros_like(loss_best_steps)

#         criterion = self.model.predict
#         criterion_indiv = self.model.predict

#         x_adv.requires_grad_()
#         grad = torch.zeros_like(x)
#         for _ in range(self.eot_iter):
#             with torch.enable_grad():
#                 loss = - criterion(x_adv).sum()
#             grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)

#         grad /= float(self.eot_iter)


#         step_size = self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1, 1])
#         x_adv_old = x_adv.clone()
#         a = 0.75
#         counter = 0
#         k = self.n_iter_2 + 0
#         u = np.arange(x.shape[0])
#         counter3 = 0

#         n_reduced = 0

#         for i in range(self.n_iter):
#             ### gradient step
#             with torch.no_grad():
#                 x_adv = x_adv.detach()
#                 grad2 = x_adv - x_adv_old
#                 x_adv_old = x_adv.clone()

#                 a = 0.75 if i > 0 else 1.0

#                 if self.norm == 'Linf':
#                     x_adv_1 = x_adv + step_size * torch.sign(grad)
#                     x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
#                     x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a), x - self.eps), x + self.eps), 0.0, 1.0)

#                 elif self.norm == 'L2':
#                     x_adv_1 = x_adv + step_size[0] * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
#                     x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
#                         self.eps * torch.ones(x.shape).to(self.device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
#                     x_adv_1 = x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a)
#                     x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
#                         self.eps * torch.ones(x.shape).to(self.device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

#                 x_adv = x_adv_1 + 0.

#         return x_adv


class APGDAttack:
    def __init__(
        self,
        model,
        n_iter=100,
        n_iter_2=22,
        n_iter_min=6,
        size_decr=3,
        norm="Linf",
        n_restarts=1,
        eps=0.3,
        show_loss=False,
        seed=0,
        loss="max_conf",
        show_acc=True,
        eot_iter=1,
        save_steps=False,
        save_dir="./results/",
        thr_decr=0.75,
        check_impr=False,
        normalize_logits=False,
        device=torch.device("cuda:0"),
        apply_softmax=True,
        classes=10,
    ):
        self.model = model
        self.n_iter = n_iter
        self.n_iter_2 = n_iter_2
        self.n_iter_min = n_iter_min
        self.size_decr = size_decr
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.show_loss = show_loss
        self.verbose = True
        self.seed = seed
        self.loss = loss
        self.show_acc = show_acc
        self.eot_iter = eot_iter
        self.save_steps = save_steps
        self.save_dir = save_dir
        self.thr_decr = thr_decr
        self.check_impr = check_impr
        self.normalize_logits = normalize_logits
        self.device = device
        self.apply_softmax = apply_softmax
        self.classes = classes

    def check_oscillation(self, x, j, k, y5, k3=0.5):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
            t += x[j - counter5] > x[j - counter5 - 1]

        return t <= k * k3 * np.ones(t.shape), t > k * 1.0 * np.ones(t.shape)

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def norm_to_interval(self, x):
        return x / (x.max(dim=1, keepdim=True)[0] + 1e-12)

    def custom_loss(self, x, y=None):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()

        return -(
            x[np.arange(x.shape[0]), y]
            - x_sorted[:, -2] * ind
            - x_sorted[:, -1] * (1.0 - ind)
        ) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    def attack_single_run(self, x_in, y_in):
        x = x_in if len(x_in.shape) == 4 else x_in.unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        if self.norm == "Linf":
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(
                self.device
            ).detach() * t / (
                t.reshape([t.shape[0], -1])
                .abs()
                .max(dim=1, keepdim=True)[0]
                .reshape([-1, 1, 1, 1])
            )
        elif self.norm == "L2":
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(
                self.device
            ).detach() * t / ((t**2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e12)
        x_adv = x_adv.clamp(0.0, 1.0)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]])
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)

        if self.loss == "ce":
            criterion = nn.CrossEntropyLoss(size_average=False)
            criterion_indiv = nn.CrossEntropyLoss(reduce=False, reduction="none")
        elif self.loss == "kl_div":
            criterion = nn.KLDivLoss(size_average=False)
            criterion_indiv = nn.KLDivLoss(reduce=False, reduction="none")
        elif self.loss == "rand_class":
            criterion = RandClassLoss(y_in)
            y_target = criterion.y_target
            criterion_indiv = RandClassLoss(y_in, y_target=y_target, reduction="none")
        elif self.loss == "max_conf":
            criterion = MaxConf(
                y_in, apply_softmax=self.apply_softmax, classes=self.classes
            )
            criterion_indiv = MaxConf(
                y_in,
                reduction="none",
                apply_softmax=self.apply_softmax,
                classes=self.classes,
            )
        elif self.loss == "last_conf":
            criterion = LastConf(y_in)
            criterion_indiv = LastConf(y_in, reduction="none")
        # elif self.loss =='custom':
        #    criterion_indiv = self.custom_loss

        ### YHLEE ###
        else:
            criterion_indiv = loss
        #############

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                if self.loss == "kl_div":
                    loss = criterion(
                        F.log_softmax(self.model(x_adv), dim=1),
                        F.softmax(self.model(x), dim=1),
                    )
                    logits = self.model(x_adv)  # 1 forward pass (eot_iter = 1)
                else:
                    if not self.normalize_logits:

                        ### YHLEE ###
                        # logits = self.model(x_adv) # 1 forward pass (eot_iter = 1)
                        # loss_indiv = criterion_indiv(logits, y)
                        loss_indiv = criterion_indiv(x_adv)
                        #############

                        loss = loss_indiv.sum()
                    else:
                        loss = self.custom_loss(self.model(x_adv), y).sum()

            grad += torch.autograd.grad(loss, [x_adv])[
                0
            ].detach()  # 1 backward pass (eot_iter = 1)

        grad /= float(self.eot_iter)

        # acc = logits.detach().max(1)[1] == y
        # acc_steps[0] = acc + 0

        loss_best = loss_indiv.detach().clone()
        loss = loss_best.sum()

        step_size = (
            self.eps
            * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach()
            * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1, 1])
        )
        x_adv_old = x_adv.clone()
        a = 0.75
        counter = 0
        k = self.n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0

        for i in range(self.n_iter):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == "Linf":
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(
                        torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps),
                        0.0,
                        1.0,
                    )
                    x_adv_1 = torch.clamp(
                        torch.min(
                            torch.max(
                                x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                                x - self.eps,
                            ),
                            x + self.eps,
                        ),
                        0.0,
                        1.0,
                    )

                elif self.norm == "L2":
                    x_adv_1 = x_adv + step_size[0] * grad / (
                        (grad**2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
                    )
                    x_adv_1 = torch.clamp(
                        x
                        + (x_adv_1 - x)
                        / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                            + 1e-12
                        )
                        * torch.min(
                            self.eps * torch.ones(x.shape).to(self.device).detach(),
                            ((x_adv_1 - x) ** 2)
                            .sum(dim=(1, 2, 3), keepdim=True)
                            .sqrt(),
                        ),
                        0.0,
                        1.0,
                    )
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(
                        x
                        + (x_adv_1 - x)
                        / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                            + 1e-12
                        )
                        * torch.min(
                            self.eps * torch.ones(x.shape).to(self.device).detach(),
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                            + 1e-12,
                        ),
                        0.0,
                        1.0,
                    )

                x_adv = x_adv_1 + 0.0

            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    if self.loss == "kl_div":
                        loss = criterion(
                            F.log_softmax(self.model(x_adv), dim=1),
                            F.softmax(self.model(x), dim=1),
                        )
                        logits = self.model(x_adv)
                    else:
                        if not self.normalize_logits:
                            ### YHLEE ###
                            # logits = self.model(x_adv) # 1 forward pass (eot_iter = 1)
                            # loss_indiv = criterion_indiv(logits, y)
                            loss_indiv = criterion_indiv(x_adv)
                            #############

                            loss = loss_indiv.sum()
                        else:
                            loss = self.custom_loss(self.model(x_adv), y).sum()

                grad += torch.autograd.grad(loss, [x_adv])[
                    0
                ].detach()  # 1 backward pass (eot_iter = 1)

            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            x_best_adv[(pred == 0).nonzero().squeeze()] = (
                x_adv[(pred == 0).nonzero().squeeze()] + 0.0
            )
            if self.show_loss:
                print(
                    "iteration: {} - Best loss: {:.6f} - Step size: {:.4f} - Reduced: {:.0f}".format(
                        i, loss_best.sum(), step_size.mean(), n_reduced
                    )
                )

            ### check step size
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1.cpu() + 0
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1

                if counter3 == k:
                    fl_oscillation, _ = self.check_oscillation(
                        loss_steps.detach().cpu().numpy(),
                        i,
                        k,
                        loss_best.detach().cpu().numpy(),
                        k3=self.thr_decr,
                    )

                    if self.check_impr:
                        fl_reduce_no_impr = (~reduced_last_check) * (
                            loss_best_last_check.cpu().numpy()
                            >= loss_best.cpu().numpy()
                        )
                        fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                        reduced_last_check = np.copy(fl_oscillation)
                        loss_best_last_check = loss_best.clone()

                    if np.sum(fl_oscillation) > 0:
                        step_size[u[fl_oscillation]] /= 2.0
                        n_reduced = fl_oscillation.astype(float).sum()

                        fl_oscillation = np.where(fl_oscillation)

                        x_adv[fl_oscillation] = x_best[fl_oscillation].clone()

                        x_new = x_best[fl_oscillation].clone().requires_grad_()
                        y_new = y[fl_oscillation].clone()
                        with torch.enable_grad():
                            grad_new = torch.zeros_like(x_new)
                            for _ in range(self.eot_iter):
                                if self.loss == "kl_div":
                                    raise ValueError("not implemented yet")
                                else:
                                    if not self.normalize_logits:
                                        logits = self.model(
                                            x_new
                                        )  # 1 forward pass (eot_iter = 1)
                                        loss_indiv = criterion_indiv(logits, y_new)
                                        loss = loss_indiv.sum()
                                    else:
                                        loss = self.custom_loss(
                                            self.model(x_new), y_new
                                        ).sum()

                            grad_new += torch.autograd.grad(loss, [x_new])[
                                0
                            ].detach()  # 1 backward pass (eot_iter = 1)
                        grad[fl_oscillation] = grad_new / float(self.eot_iter)

                    counter3 = 0
                    k = np.maximum(k - self.size_decr, self.n_iter_min)

        ### save intermediate steps
        if self.save_steps:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            return acc_steps, loss_best_steps

            torch.save(
                {"acc_steps": acc_steps, "loss_steps": loss_best_steps},
                self.save_dir
                + "/apgd_singlestep_{}_eps_{:.5f}_niter_{:.0f}_thrdecr_{:.2}.pth".format(
                    self.norm, self.eps, self.n_iter, self.thr_decr
                ),
            )
            scipy.io.savemat(
                self.save_dir
                + "/apgd_singlestep_{}_eps_{:.5f}_niter_{:.0f}_thrdecr_{:.2}.pth".format(
                    self.norm, self.eps, self.n_iter, self.thr_decr
                ),
                {
                    "acc_steps": acc_steps.cpu().detach().numpy(),
                    "loss_steps": loss_best_steps.cpu().detach().numpy(),
                },
            )

        return x_best, acc, loss_best, x_best_adv

    def perturb(self, x, y, best_loss=False, cheap=True):
        assert self.norm in ["Linf", "L2"]

        adv = x.clone()
        ### YHLEE ###
        # acc = self.model(x).max(1)[1] == y
        # loss = -1e10 * torch.ones_like(acc).float()
        # if self.show_acc:
        #     print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(self.norm, self.eps))
        #     print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        # startt = time.time()
        #############

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)

        if self.save_steps:
            assert self.n_restarts == 1
            acc, loss = self.attack_single_run(x, y)

            return acc, loss

        if not cheap:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(self.device) * (-float("inf"))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.0
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.0

                if self.verbose:
                    print("restart {} - loss: {:.5f}".format(counter, loss_best.sum()))

            adv = adv_best

        ### YHLEE ###
        # else:
        #     for counter in range(self.n_restarts):
        #         ind_to_fool = acc.nonzero().squeeze()
        #         if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
        #         if ind_to_fool.numel() != 0:
        #             x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
        #             best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
        #             ind_curr = (acc_curr == 0).nonzero().squeeze()
        #             #acc_temp = torch.zeros_like(acc)
        #             acc[ind_to_fool[ind_curr]] = 0
        #             adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
        #             if self.show_acc: print('restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s'.format(counter, acc.float().mean(), time.time() - startt))
        ############

        # y_pred = (self.model(adv).max(1)[1] == y).float()
        # print(y_pred.mean()*100, (adv - x).abs().reshape([-1]).max(0)[0])

        return adv  # , None, None
