import torch
import numpy as np

'''eval tools'''
def attack(m, method, x, device):
    eps = 0.01
    iterations = 200
    stepsize = 0.1
    budget = 5
    n_restarts = 5
    
    noise = DeContraster(eps)
    attack1 = modified_MonotonePGD(m, eps, iterations, stepsize, momentum=0.9, 
                norm='inf', normalize_grad=False, restarts=0,
                init_noise_generator=noise, save_trajectory=False)
    
    noise = UniformNoiseGenerator(min=-eps, max=eps)
    attack2 = modified_MonotonePGD(m, eps, iterations, stepsize, momentum=0.9, 
                norm='inf', normalize_grad=False, restarts=0,
                init_noise_generator=noise, save_trajectory=False)

    noise = NormalNoiseGenerator(sigma=1e-4)
    attack3 = modified_MonotonePGD(m, eps, iterations, stepsize, momentum=0.9, 
                norm='inf', normalize_grad=False, restarts=0,
                init_noise_generator=noise, save_trajectory=False)
    
    attack = APGDAttack(
        m, n_iter=100*budget, n_iter_2=22*budget, n_iter_min=6*budget, 
        size_decr=3, norm='Linf', eps=eps, seed=0, eot_iter=1, thr_decr=.75, 
        check_impr=False, device=device) 
    
    list_pred = []
    for att in [attack1, attack2, attack3]:
        x_adv = att.perturb(x)
        pred = method(x_adv)
        list_pred.append(pred.unsqueeze(0))
        
    for _ in range(n_restarts):
        x_adv = attack.perturb(x)
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
    **kwargs
):
    method = getattr(m, method)
    l_result = []
    for batch in dl:
        if input_type == "first":
            x = batch[0]

        if no_grad:
            with torch.no_grad():
                if flatten:
                    x = x.view(len(x), -1)
                pred = attack(m, method, x.cuda(device), device=device).detach().cpu()
        else:
            if flatten:
                x = x.view(len(x), -1)
            pred = attack(m, method, x.cuda(device), device=device).detach().cpu()

        l_result.append(pred)
    return torch.cat(l_result)

'''PGD attacks tools'''
def normalize_perturbation(perturbation, p):
    if p == 'inf':
        return perturbation.sign()
    elif p==2 or p==2.0:
        bs = perturbation.shape[0]
        pert_flat = perturbation.view(bs, -1)
        pert_normalized = torch.nn.functional.normalize(pert_flat, p=p, dim=1)
        return pert_normalized.view_as(perturbation)
    else:
        raise NotImplementedError('Projection only supports l2 and inf norm')

def project_perturbation(perturbation, eps, p):
    if p == 'inf':
        mask = perturbation.abs() > eps
        pert_normalized = perturbation
        pert_normalized[mask] = eps * perturbation[mask].sign()
        return pert_normalized
    elif p==2 or p==2.0:
        #TODO use torch.renorm
        bs = perturbation.shape[0]
        pert_flat = perturbation.view(bs, -1)
        norm = torch.norm(perturbation.view(bs, -1), dim=1) + 1e-10
        mask = norm > eps
        pert_normalized = pert_flat
        pert_normalized[mask, :] = (eps / norm[mask, None]) * pert_flat[mask, :]
        return pert_normalized.view_as(perturbation)
    else:
        raise NotImplementedError('Projection only supports l2 and inf norm')

def calculate_smart_lr(prev_mean_lr, lr_accepted, lr_decay, iterations, max_lr):
    accepted_idcs = lr_accepted > 0
    if torch.sum(accepted_idcs).item() > 0:
        new_lr = 0.5 * (prev_mean_lr + torch.mean(lr_accepted[lr_accepted > 0]).item())
    else:
        new_lr = prev_mean_lr * ( lr_decay ** iterations )

    new_lr = min(max_lr, new_lr)
    return new_lr

class modified_MonotonePGD():
    def __init__(self, model, eps, iterations, stepsize, momentum=0.9, 
                 lr_smart=False, lr_decay=0.5, lr_gain=1.1,
                 norm='inf', normalize_grad=False, restarts=0,
                 init_noise_generator=None, save_trajectory=False):
        super().__init__()
        #loss either pass 'CrossEntropy' or 'LogitsDiff' or custom loss function
        self.eps = eps
        self.iterations = iterations
        self.stepsize = stepsize
        self.momentum = momentum
        self.norm = norm
        self.normalize_grad = normalize_grad
        self.init_noise_generator = init_noise_generator
        self.lr_smart = lr_smart
        self.prev_mean_lr = stepsize
        self.lr_decay = lr_decay #stepsize decay
        self.lr_gain = lr_gain
        self.restarts = restarts
        self.save_trajectory = False
        self.last_trajectory = None
        self.model = model
    
    def check_model(self):
        if self.model is None:
            raise RuntimeError('Attack model not set')
            
    def __get_trajectory_depth(self):
        return self.iterations + 1

    def perturb_inner(self, x, y=None, targeted=False):
        l_f = self.model.predict(x)
        
        if self.lr_smart:
            lr_accepted = -1 * x.new_ones(self.iterations, x.shape[0])
            lr = self.prev_mean_lr * x.new_ones(x.shape[0])
        else:
            lr = self.stepsize * x.new_ones(x.shape[0])

        #initialize perturbation
        if self.init_noise_generator is None:
            pert = torch.zeros_like(x)
        else:
            pert = self.init_noise_generator(x)
            pert = torch.clamp(x + pert, 0, 1) - x  # box constraint
            pert = project_perturbation(pert, self.eps, self.norm)

        #TODO fix the datatype here !!!
        prev_loss = 1e13 * x.new_ones(x.shape[0], dtype=torch.float)

        prev_pert = pert.clone().detach()
        prev_velocity = torch.zeros_like(pert)
        velocity = torch.zeros_like(pert)

        #trajectory container
        if self.save_trajectory:
            trajectory = torch.zeros((self.iterations + 1,) + x.shape, device=x.device)
            trajectory[0, :] = x
        else:
            trajectory = None

        for i in range(self.iterations + 1):
            pert.requires_grad_(True)
            with torch.enable_grad():
                data = x + pert
                loss_expanded = self.model.predict(data)
                loss = torch.mean(loss_expanded)
                grad = torch.autograd.grad(loss, pert)[0]

            with torch.no_grad():
                loss_increase_idx = loss_expanded > prev_loss

                pert[loss_increase_idx, :] = prev_pert[loss_increase_idx, :].clone().detach()
                loss_expanded[loss_increase_idx] = prev_loss[loss_increase_idx].clone().detach()
                prev_pert = pert.clone().detach()
                prev_loss = loss_expanded
                #previous velocity always holds the last accepted velocity vector
                #velocity the one used for the last update that might have been rejected
                velocity[loss_increase_idx, :] = prev_velocity[loss_increase_idx, :]
                prev_velocity = velocity.clone().detach()

                if i > 0:
                    #use standard lr in firt iteration
                    lr[loss_increase_idx] *= self.lr_decay
                    lr[~loss_increase_idx] *= self.lr_gain

                if i == self.iterations:
                    break

                if self.lr_smart:
                    lr_accepted[i, ~loss_increase_idx] = lr[~loss_increase_idx]

                #pgd on given loss
                if self.normalize_grad:
                    if self.momentum > 0:
                        #https://arxiv.org/pdf/1710.06081.pdf the l1 normalization follows the momentum iterative method
                        l1_norm_gradient = 1e-10 + torch.sum(grad.abs().view(x.shape[0], -1), dim=1).view(-1,1,1,1)
                        velocity = self.momentum * velocity + grad / l1_norm_gradient
                    else:
                        velocity = grad
                    norm_velocity = normalize_perturbation(velocity, self.norm)
                else:
                    # velocity update as in pytorch https://pytorch.org/docs/stable/optim.html
                    velocity = self.momentum * velocity + grad
                    norm_velocity = velocity

                pert = pert - lr[:,None,None,None] * norm_velocity
                pert = project_perturbation(pert, self.eps, self.norm)
                pert = torch.clamp(x + pert, 0, 1) - x #box constraint

                if self.save_trajectory:
                    trajectory[i + 1] = x + pert

        if self.lr_smart:
            self.prev_mean_lr = calculate_smart_lr(self.prev_mean_lr, lr_accepted, self.lr_decay, self.iterations, 2 * self.eps)

        return data, loss_expanded, trajectory
    
    def perturb(self, x, y=None, targeted=False):
        #base class method that handles various restarts
        self.check_model()
        self.model.eval()

        restarts_data = x.new_empty((1 + self.restarts,) + x.shape)
        restarts_objs = x.new_empty((1 + self.restarts, x.shape[0]))

        if self.save_trajectory:
            self.last_trajectory = None
            trajectories_shape = (self.restarts,) + (self.__get_trajectory_depth(),) + x.shape
            restart_trajectories = x.new_empty(trajectories_shape)

        for k in range(1 + self.restarts):
            k_data, k_obj, k_trajectory = self.perturb_inner(x, y=None, targeted=targeted)
            restarts_data[k, :] = k_data
            restarts_objs[k, :] = k_obj
            if self.save_trajectory:
                restart_trajectories[k, :] = k_trajectory

        bs = x.shape[0]
        best_idx = torch.argmin(restarts_objs, 0)
        best_data = restarts_data[best_idx, range(bs), :]

        if self.save_trajectory:
            self.last_trajectory = restart_trajectories[best_idx, :, range(bs), :]

        return best_data
    
class AdversarialNoiseGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, x):
        #generate noise matching the size of x
        raise NotImplementedError()
    
class DeContraster(AdversarialNoiseGenerator):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        diff = torch.clamp(x.mean(dim=(1,2,3))[:,None,None,None] - x, -self.eps, self.eps)
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
    
'''APGD attacks tools'''
class APGDAttack():
    def __init__(self, model, n_iter=100, n_iter_2=22, n_iter_min=6, size_decr=3,
                 norm='Linf', eps=0.3, show_loss=False, seed=0,
                 eot_iter=1, thr_decr=.75, check_impr=False,
                  device=torch.device('cuda:0')):
        self.model = model
        self.n_iter = n_iter
        self.n_iter_2 = n_iter_2
        self.n_iter_min = n_iter_min
        self.size_decr = size_decr
        self.eps = eps
        self.norm = norm
        self.show_loss = show_loss    
        self.verbose = True
        self.seed = seed
        self.eot_iter = eot_iter
        self.thr_decr = thr_decr
        self.check_impr = check_impr
        self.device = device
    
    def check_oscillation(self, x, j, k, y5, k3=0.5):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
            t += x[j - counter5] > x[j - counter5 - 1]
          
        return t <= k*k3*np.ones(t.shape), t > k*1.0*np.ones(t.shape)
        
    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)
    
    def norm_to_interval(self, x):
        return x / (x.max(dim=1, keepdim=True)[0] + 1e-12)
        
    
    def custom_loss(self, x, y=None):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        
        return -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
        
    def perturb(self, x_in):
        x = x_in if len(x_in.shape) == 4 else x_in.unsqueeze(0)
        
        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e12)
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]])
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)
        
        criterion = self.model.predict
        criterion_indiv = self.model.predict
        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                loss = - criterion(x_adv).sum()
            grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            
        grad /= float(self.eot_iter)
        
        
        step_size = self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1, 1])
        x_adv_old = x_adv.clone()
        a = 0.75
        counter = 0
        k = self.n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0
        
        n_reduced = 0
        
        for i in range(self.n_iter):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()
                
                a = 0.75 if i > 0 else 1.0
                
                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a), x - self.eps), x + self.eps), 0.0, 1.0)
                    
                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size[0] * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)
                    
                x_adv = x_adv_1 + 0.
              
        return x_adv