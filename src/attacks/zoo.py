import numpy as np
import torch


def zoo_attack(x0, detector, const, max_iter=1000, max_l2=3):

    D = 3 * 32 * 32
    D = np.prod(x0.shape[1:])
    real_modifier = torch.zeros(1, 3, 32, 32)
    batch_size = 64
    modifier_up = np.zeros(D, dtype=np.float32)
    modifier_down = np.zeros(D, dtype=np.float32)
    mt = np.zeros(D, dtype=np.float32)
    vt = np.zeros(D, dtype=np.float32)
    adam_epoch = np.ones(D, dtype=np.int32)
    grad = np.zeros(batch_size, dtype=np.float32)
    hess = np.zeros(batch_size, dtype=np.float32)

    max_iter = 200

    h = 0.0001
    const = 0.1
    step_size = 0.01
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    modifier_up = 1 - x0.clone().detach().view(-1).cpu().numpy()
    modifier_down = 0 - x0.clone().detach().view(-1).cpu().numpy()

    for i_iter in tqdm(range(max_iter)):
        losses, l2_loss, adv_loss, scores, pert_image = loss_run(
            x0, real_modifier, const
        )
        #     print(losses, l2_loss, adv_loss)
        var_list = np.array(np.arange(D), dtype=np.int32)
        indices = var_list[np.random.choice(var_list.size, batch_size, replace=False)]
        var = real_modifier.clone().repeat((2 * batch_size + 1, 1, 1, 1))
        for i in range(batch_size):
            var[i * 2 + 1].view(-1)[indices[i]] += h
            var[i * 2 + 2].view(-1)[indices[i]] -= h
        losses, l2_loss, adv_loss, scores, pert_images = loss_run(x0, var, const)
        real_modifier_numpy = real_modifier.detach().cpu().numpy()
        coordinate_ADAM(
            losses.detach().cpu().numpy(),
            indices,
            grad,
            hess,
            batch_size,
            mt,
            vt,
            real_modifier_numpy,
            adam_epoch,
            modifier_up,
            modifier_down,
            step_size,
            adam_beta1,
            adam_beta2,
            proj=True,
        )
        real_modifier = torch.from_numpy(real_modifier_numpy)

    return {"pert_image": pert_iamge}


def loss_run(x, detector, real_modifier, const):
    pert_out = x + real_modifier
    output = detector.predict(pert_out)
    l2_loss = ((pert_out - x) ** 2).sum()
    adv_loss = -output
    loss = l2_loss + const * adv_loss
    return (
        loss.detach(),
        l2_loss.detach(),
        adv_loss.detach(),
        output.detach(),
        pert_out.detach(),
    )
