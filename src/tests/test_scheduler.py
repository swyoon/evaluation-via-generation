from torch.optim import Adam

from models.modules import ConvNet2
from schedulers import get_scheduler


def test_warmup():
    net = ConvNet2()  # dummy model
    opt = Adam(net.parameters(), lr=1e-4)  # dummy optimizer
    d_sch = {"name": "warmup", "warmup": 10, "verbose": True}
    sch = get_scheduler(opt, d_sch)
    opt.step()
    sch.step()
