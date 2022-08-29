import torch
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Normalize

from attacks import Detector, get_advdist
from attacks.advdist import (
    AdversarialDistribution,
    AdversarialDistributionAE,
    AdversarialDistributionTransform,
)
from attacks.mcmc import MHSampler
from loader.synthetic import energy2d
from models import load_pretrained
from models.ae import AE
from models.cls import ResNetClassifier
from models.glow.models import GlowV2
from models.modules import ConvNet3FCBN, DeConvNet3


class DummyDetector:
    def predict(self, x):
        return x.flatten(1).sum(-1)


def test_adv_dist_glow():
    device = "cpu"
    model = GlowV2(
        (1, 1, 2),
        hidden_channels=512,
        K=32,
        L=1,
        actnorm_scale=1.0,
        flow_permutation="invconv",
        flow_coupling="affineV2",
        vector_mode=True,
    )
    """initialize actnorm parameter by forwarding arbitrary data"""
    model(x=torch.randn(50, 2, 1, 1, dtype=torch.float), reverse=False)
    ad = AdversarialDistribution(model, T=1.0, mode="reverseKL", flow=True)
    ad.to(device)
    opt = Adam(ad.parameters(), lr=1e-4)
    dummy_detector = DummyDetector()
    detector = Detector(dummy_detector)
    batch_size = 50
    d_train = ad.train_step(batch_size, device, detector, opt)


def test_detector():
    model = DummyDetector()
    x = torch.randn(10, 2, dtype=torch.float)
    y = torch.randn(10, 1)  # dummy y variable

    def normalize(x):
        return (x - 1) / 0.5

    detector = Detector(model, transform=normalize)

    ds = TensorDataset(x, y)
    dl = DataLoader(ds)
    detector.learn_normalization(dataloader=dl)
    print(detector.mean, detector.std)

    score = detector.predict(x)
    assert score.shape == (10,)
    assert (score.mean() - 0.0) ** 2 < 1e-2


def test_detector_rank():
    model = DummyDetector()
    x = torch.randn(10, 2, dtype=torch.float)
    y = torch.randn(10, 1)  # dummy y variable

    def normalize(x):
        return (x - 1) / 0.5

    detector = Detector(model, transform=normalize, use_rank=True)

    ds = TensorDataset(x, y)
    dl = DataLoader(ds)
    detector.learn_normalization(dataloader=dl)
    print(detector.mean, detector.std)

    score = detector.predict(x)
    assert score.min() >= -1
    assert score.max() <= 1


# def test_from_cfg():
#     s_cfg = """
#         advdist:
#           name: advq
#           T: 1.
#           classifier_thres: 0.9
#           model:
#             arch: rnnlm
#             input_shape: [4, 4]
#             hidden_size: 128
#             K: 10
#           vqvae:
#             identifier: mnist32fmnist32_ood_vqvae/K10_4x4
#             config_file: K10_4x4.yml
#             ckpt_file: model_epoch_280.pkl
#           classifier:
#             identifier: fmnist_v_mnist/res18_lr5
#             config_file: res18.yml
#             ckpt_file: model_best.pkl
#         """
#     cfg = OmegaConf.create(s_cfg)
#     advdist = get_advdist(cfg)
#     d_sample = advdist.sample(n_sample=5, device='cpu', reject=True)


def test_adv_dist_ae():
    # detector
    detector = Detector(DummyDetector())

    # ae
    encoder = ConvNet3FCBN(in_chan=1, out_chan=2)
    decoder = DeConvNet3(in_chan=2, out_chan=1)
    ae = AE(encoder, decoder)

    # glow
    model = GlowV2(
        (1, 1, 2),
        hidden_channels=32,
        K=32,
        L=1,
        actnorm_scale=1.0,
        flow_permutation="invconv",
        flow_coupling="affineV2",
        vector_mode=True,
    )

    """initialize actnorm parameter by forwarding arbitrary data"""
    model(x=torch.randn(50, 2, 1, 1, dtype=torch.float), reverse=False)

    ## advdist
    advdist = AdversarialDistributionAE(model, ae, T=0.1, barrier="norm")

    ## train
    opt = Adam(advdist.model.parameters(), lr=1e-4)
    d_train = advdist.train_step(10, "cpu", detector, opt)


def test_adv_dist_mh():
    # detector
    detector = Detector(DummyDetector())
    classifier = ResNetClassifier()

    # ae
    encoder = ConvNet3FCBN(in_chan=3, out_chan=2)
    decoder = DeConvNet3(in_chan=2, out_chan=3)
    ae = AE(encoder, decoder)

    advdist = AdversarialDistributionAE(
        model="mh",
        ae=ae,
        T=0.1,
        classifier=classifier,
        classifier_thres_logit=-1,
        z_shape=(2, 1, 1),
        n_step=2,
        detector=detector,
        stepsize=0.1,
        z_bound=1,
        classifier_mean=[0.4914, 0.4822, 0.4465],
        classifier_std=[0.2023, 0.1994, 0.2010],
    )
    d_result = advdist.sample(n_sample=10, device="cpu")
    z0 = torch.rand(10, 2, 1, 1, dtype=torch.float)
    d_result = advdist.sample(z0=z0)


# def test_md_advdist():
#     # detector
#     md, _ = load_pretrained('cifar_ood_md/md_resnet', 'md_resnet_cifar.yml',
#                                'resnet_cifar10.pth', lr_tunned_with='SVHN_OOD', device='cpu')
#     detector = Detector(md)
#
#     # ae
#     encoder = ConvNet3FCBN(in_chan=3, out_chan=2)
#     decoder = DeConvNet3(in_chan=2, out_chan=3)
#     ae = AE(encoder, decoder)
#
#     # glow
#     model = GlowV2(
#         (1,1,2),
#         hidden_channels=32,
#         K=32,
#         L=1,
#         actnorm_scale=1.,
#         flow_permutation='invconv',
#         flow_coupling='affineV2',
#         vector_mode=True)
#
#     '''initialize actnorm parameter by forwarding arbitrary data'''
#     model(x=torch.randn(50,2,1,1, dtype=torch.float), reverse=False)
#
#     ## advdist
#     advdist = AdversarialDistributionAE(model, ae, T=0.1, barrier='norm')
#
#     ## train
#     opt = Adam(advdist.model.parameters(), lr=1e-4)
#     d_train = advdist.train_step(10, 'cpu', detector, opt)


def test_advdist_transform():
    detector = Detector(DummyDetector())
    sampler = MHSampler(n_step=5, stepsize=0.1, bound=(0, 1), T=1.0, sample_shape=(5,))
    advdist = AdversarialDistributionTransform(
        detector=detector, transform="affineV0", sampler=sampler, z_bound=(0, 1)
    )
    img = torch.rand(10, 3, 32, 32)
    d_sample = advdist.sample(img)
    assert "min_x" in d_sample
    assert "min_E" in d_sample
    assert "min_img" in d_sample


def test_adtr_from_cfg():
    detector = Detector(DummyDetector())
    s_cfg = """
        advdist:
          name: adtr
          transform: colorV0
          z_bound: [0, 1]
          sampler:
            name: mh
            n_step: 3
            stepsize: 0.1
            bound: [0, 1]
            T: 0.1
            sample_shape: [4]
        """
    cfg = OmegaConf.create(s_cfg)
    advdist = get_advdist(cfg)
    advdist.detector = detector
    img = torch.rand(2, 3, 32, 32)
    d_sample = advdist.sample(img)
    assert "min_x" in d_sample
    assert "min_E" in d_sample


def test_ad_linf_from_cfg():
    detector = Detector(DummyDetector())
    s_cfg = """
        advdist:
          name: adlinf
          sampler:
            name: coord
            h: 0.1
            stepsize: 0.1
            Linf: 0.01
            half_every: 2000
            momentum: 0.909
            n_step: 3
            bound: [0, 1]
        """
    cfg = OmegaConf.create(s_cfg)
    advdist = get_advdist(cfg)
    advdist.detector = detector
    img = torch.rand(2, 3, 32, 32)
    d_sample = advdist.sample(img)
    assert "min_x" in d_sample
    assert "min_E" in d_sample


def test_ad_stylegan2_cfg():
    detector = Detector(DummyDetector())
    s_cfg = """
        advdist:
          name: adstylegan2
          stylegan2_g:
            arch: stylegan2_g
            identifier: svhn_stylegan2/z64
            config_file: generator.yml
            ckpt_file: model=G_ema-best-weights-step=188000.pth
          sampler:
            name: mh
            n_step: 3
            stepsize: 0.1
            bound: spherical
            T: 0.1
            sample_shape: [64]
            initial_dist: uniform_sphere
        device: cpu
        """
    cfg = OmegaConf.create(s_cfg)
    advdist = get_advdist(cfg)
    advdist.detector = detector
    d_sample = advdist.sample(n_sample=2, device="cpu")
    assert "min_x" in d_sample
    assert "min_E" in d_sample
