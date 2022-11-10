import copy
import os

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from omegaconf import OmegaConf

import models.CSI.models.classifier as C
import models.CSI.models.transform_layers as TL
from augmentations import get_composed_augmentations
from loader import get_dataloader
from models.ae import AE, DAE, VAE, WAE
from models.cls import ResNetClassifier
from models.CSI.common.common import parse_args
from models.CSI.datasets import get_dataset, get_subclass_dataset, get_superclass_list
from models.CSI.utils.utils import normalize, set_random_seed
from models.energybased import EnergyBasedModel
from models.glow.models import GlowV2
from models.glow_y0ast.models import Glow_y0ast
from models.igebm import IGEBM
from models.mcmc import get_sampler
from models.MD.densenet import DenseNet3 as MDDenseNet
from models.MD.md import MD
from models.MD.resnet import ResNet34 as MDResNet34
from models.modules import (
    ConvDenoiser,
    ConvMLP,
    ConvNet2,
    ConvNet2Att,
    ConvNet2FC,
    ConvNet2p,
    ConvNet3,
    ConvNet3Att,
    ConvNet3AttV2,
    ConvNet3FC,
    ConvNet3FCBN,
    ConvNet3MLP,
    ConvNet64,
    DCGANDecoder,
    DCGANDecoderNoBN,
    DCGANEncoder,
    DeConvNet2,
    DeConvNet3,
    DeConvNet64,
    FCNet,
    FCResNet,
    IGEBMEncoder,
    ResNet1x1,
)
from models.modules_sngan import Generator as SNGANGeneratorBN
from models.modules_sngan import GeneratorGN as SNGANGeneratorGN
from models.modules_sngan import GeneratorNoBN as SNGANGeneratorNoBN
from models.modules_sngan import GeneratorNoBN64 as SNGANGeneratorNoBN64
from models.nae import (
    NAE,
    NAE_CL_CD,
    NAE_CL_NCE,
    NAE_CL_OMI,
    NAE_CLX_OMI,
    NAE_CLZX_OMI,
    NAE_L2_OMI,
)
from models.OE.allconv import AllConvNet
from models.OE.outlier_exposure import OutlierExposure
from models.OE.wrn import WideResNet
from models.pixelcnn import PixelCNN_OC

# from models.ViT import load_pretrained_vit_tf
from models.ViT_HF import load_pretrained_vit_hf

def get_net(in_dim, out_dim, **kwargs):
    nh = kwargs.get("nh", 8)
    out_activation = kwargs.get("out_activation", "linear")

    if kwargs["arch"] == "conv2":
        net = ConvNet2(
            in_chan=in_dim, out_chan=out_dim, nh=nh, out_activation=out_activation
        )
    elif kwargs["arch"] == "conv2p":
        net = ConvNet2p(
            in_chan=in_dim, out_chan=out_dim, nh=nh, out_activation=out_activation
        )
    elif kwargs["arch"] == "conv2fc":
        nh_mlp = kwargs["nh_mlp"]
        net = ConvNet2FC(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            nh_mlp=nh_mlp,
            out_activation=out_activation,
            use_spectral_norm=kwargs.get("use_spectral_norm", False),
        )
    elif kwargs["arch"] == "conv2att":
        resdim = kwargs["resdim"]
        n_res = kwargs["n_res"]
        ver = kwargs["ver"]
        net = ConvNet2Att(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            resdim=resdim,
            n_res=n_res,
            ver=ver,
            out_activation=out_activation,
        )

    elif kwargs["arch"] == "deconv2":
        net = DeConvNet2(
            in_chan=in_dim, out_chan=out_dim, nh=nh, out_activation=out_activation
        )
    elif kwargs["arch"] == "conv64":
        num_groups = kwargs.get("num_groups", None)
        use_bn = kwargs.get("use_bn", False)
        net = ConvNet64(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            out_activation=out_activation,
            num_groups=num_groups,
            use_bn=use_bn,
        )
    elif kwargs["arch"] == "deconv64":
        num_groups = kwargs.get("num_groups", None)
        use_bn = kwargs.get("use_bn", False)
        net = DeConvNet64(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            out_activation=out_activation,
            num_groups=num_groups,
            use_bn=use_bn,
        )
    elif kwargs["arch"] == "conv3":
        net = ConvNet3(
            in_chan=in_dim, out_chan=out_dim, nh=nh, out_activation=out_activation
        )
    elif kwargs["arch"] == "conv3fc":
        nh_mlp = kwargs["nh_mlp"]
        net = ConvNet3FC(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            nh_mlp=nh_mlp,
            out_activation=out_activation,
        )
    elif kwargs["arch"] == "conv3fcbn":
        nh_mlp = kwargs["nh_mlp"]
        encoding_range = kwargs.get("encoding_range", None)
        net = ConvNet3FCBN(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            nh_mlp=nh_mlp,
            out_activation=out_activation,
            encoding_range=encoding_range,
        )

    elif kwargs["arch"] == "conv3mlp":
        l_nh_mlp = kwargs["l_nh_mlp"]
        activation = kwargs["activation"]
        net = ConvNet3MLP(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            l_nh_mlp=l_nh_mlp,
            out_activation=out_activation,
            activation=activation,
        )
    elif kwargs["arch"] == "conv3att":
        resdim = kwargs["resdim"]
        n_res = kwargs["n_res"]
        ver = kwargs["ver"]
        activation = kwargs["activation"]
        net = ConvNet3Att(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            resdim=resdim,
            n_res=n_res,
            ver=ver,
            out_activation=out_activation,
            activation=activation,
        )
    elif kwargs["arch"] == "conv3attV2":
        resdim = kwargs["resdim"]
        activation = kwargs["activation"]
        spherical = kwargs["spherical"]
        net = ConvNet3AttV2(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            resdim=resdim,
            out_activation=out_activation,
            activation=activation,
            spherical=spherical,
        )

    elif kwargs["arch"] == "deconv3":
        num_groups = kwargs.get("num_groups", None)
        net = DeConvNet3(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            out_activation=out_activation,
            num_groups=num_groups,
        )
    elif kwargs["arch"] == "fc":
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        net = FCNet(
            in_dim=in_dim,
            out_dim=out_dim,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
            flatten_input=kwargs.get("flatten_input", False),
        )
    elif kwargs["arch"] == "fcres":
        net = FCResNet(
            in_dim=in_dim,
            out_dim=out_dim,
            res_dim=kwargs["resdim"],
            n_res_hidden=kwargs["n_res_hidden"],
            n_resblock=kwargs["n_resblock"],
            out_activation=out_activation,
            use_spectral_norm=kwargs.get("use_spectral_norm", False),
            flatten_input=kwargs.get("flatten_input", False),
        )
    elif kwargs["arch"] == "res1x1":
        net = ResNet1x1(
            in_dim=in_dim,
            out_dim=out_dim,
            res_dim=kwargs["resdim"],
            n_res_hidden=kwargs["n_res_hidden"],
            n_resblock=kwargs["n_resblock"],
            out_activation=out_activation,
            use_spectral_norm=kwargs.get("use_spectral_norm", False),
        )

    elif kwargs["arch"] == "convmlp":
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        spatial_dim = kwargs.get("spatial_dim", 1)
        fusion_at = kwargs.get("fusion_at", 0)
        net = ConvMLP(
            in_dim=in_dim,
            out_dim=out_dim,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
            spatial_dim=spatial_dim,
            fusion_at=fusion_at,
        )
    elif kwargs["arch"] == "convdenoiser":
        sig = kwargs["sig"]
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        net = ConvDenoiser(
            in_dim=in_dim,
            out_dim=out_dim,
            sig=sig,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
        )
    elif kwargs["arch"] == "dcgan_encoder":
        bias = kwargs.get("bias", True)
        print(f"DCGAN encoder bias: {bias}")
        net = DCGANEncoder(in_chan=in_dim, out_chan=out_dim, bias=bias)
    elif kwargs["arch"] == "dcgan_decoder":
        bias = kwargs.get("bias", True)
        print(f"DCGAN decoder bias: {bias}")
        net = DCGANDecoder(in_chan=in_dim, out_chan=out_dim, bias=bias)
    elif kwargs["arch"] == "dcgan_decoder_nobn":
        bias = kwargs.get("bias", True)
        print(f"DCGAN decoder bias: {bias}")
        out_activation = kwargs["out_activation"]
        net = DCGANDecoderNoBN(
            in_chan=in_dim, out_chan=out_dim, bias=bias, out_activation=out_activation
        )
    elif kwargs["arch"] == "vqvae_encoder_32_8":
        net = VQVAEEncoder32_8(in_chan=in_dim, out_chan=out_dim)
    elif kwargs["arch"] == "vqvae_decoder_32_8":
        net = VQVAEDecoder32_8(in_chan=in_dim, out_chan=out_dim)
    elif kwargs["arch"] == "vqvae_encoder_32_4":
        net = VQVAEEncoder32_4(in_chan=in_dim, out_chan=out_dim)
    elif kwargs["arch"] == "vqvae_decoder_32_4":
        net = VQVAEDecoder32_4(
            in_chan=in_dim, out_chan=out_dim, out_activation=kwargs["out_activation"]
        )
    elif kwargs["arch"] == "IGEBM":
        net = IGEBM(in_chan=in_dim)
    elif kwargs["arch"] == "GPND_E":
        for_mnist = kwargs.get("mnist", False)
        net = GPND_net.Encoder(out_dim, channels=in_dim, mnist=for_mnist)
    elif kwargs["arch"] == "GPND_G":
        for_mnist = kwargs.get("mnist", False)
        if for_mnist:
            net = GPND_net.GeneratorMNIST(in_dim, channels=out_dim)
        else:
            net = GPND_net.Generator(in_dim, channels=out_dim)
    elif kwargs["arch"] == "GPND_D":
        for_mnist = kwargs.get("mnist", False)
        net = GPND_net.Discriminator(channels=in_dim, mnist=for_mnist)
    elif kwargs["arch"] == "GPND_ZD":
        net = GPND_net.ZDiscriminator(in_dim)
    elif kwargs["arch"] == "IGEBMEncoder":
        use_spectral_norm = kwargs.get("user_spectral_norm", False)
        keepdim = kwargs.get("keepdim", True)
        out_activation = kwargs.get("out_activation", "linear")
        net = IGEBMEncoder(
            in_chan=in_dim,
            out_chan=out_dim,
            n_class=None,
            use_spectral_norm=use_spectral_norm,
            keepdim=keepdim,
            out_activation=out_activation,
            avg_pool_dim=kwargs.get("avg_pool_dim", 1),
        )
    elif kwargs["arch"] == "JEMWideResNet":
        depth = kwargs.get("depth", 28)
        width = kwargs.get("width", 10)
        dropout_rate = kwargs.get("dropout_rate", 0.0)
        net = Wide_ResNet(
            depth=depth, widen_factor=width, norm=None, dropout_rate=dropout_rate
        )

    elif kwargs["arch"] == "MDDenseNet":
        depth = kwargs.get("depth", 100)
        net = MDDenseNet(depth, out_dim)

    elif kwargs["arch"] == "MDResNet34":
        num_classes = out_dim
        net = MDResNet34(num_classes)

    elif kwargs["arch"] == "OdinDenseNet":
        depth = kwargs.get("depth", 100)
        net = OdinDenseNet(depth, out_dim, in_dim)

    elif kwargs["arch"] == "OdinWideResNet":
        depth = kwargs.get("depth", 28)
        net = OdinWideResNet(depth, out_dim, in_dim)

    elif kwargs["arch"] == "sngan_generator_bn":
        hidden_dim = kwargs.get("hidden_dim", 128)
        out_activation = kwargs["out_activation"]
        net = SNGANGeneratorBN(
            z_dim=in_dim,
            channels=out_dim,
            hidden_dim=hidden_dim,
            out_activation=out_activation,
        )
    elif kwargs["arch"] == "sngan_generator_nobn":
        hidden_dim = kwargs.get("hidden_dim", 128)
        out_activation = kwargs["out_activation"]
        net = SNGANGeneratorNoBN(
            z_dim=in_dim,
            channels=out_dim,
            hidden_dim=hidden_dim,
            out_activation=out_activation,
        )
    elif kwargs["arch"] == "sngan_generator_nobn64":
        hidden_dim = kwargs.get("hidden_dim", 128)
        out_activation = kwargs["out_activation"]
        net = SNGANGeneratorNoBN64(
            z_dim=in_dim,
            channels=out_dim,
            hidden_dim=hidden_dim,
            out_activation=out_activation,
        )
    elif kwargs["arch"] == "sngan_generator_gn":
        hidden_dim = kwargs.get("hidden_dim", 128)
        out_activation = kwargs["out_activation"]
        num_groups = kwargs["num_groups"]
        net = SNGANGeneratorGN(
            z_dim=in_dim,
            channels=out_dim,
            hidden_dim=hidden_dim,
            out_activation=out_activation,
            num_groups=num_groups,
            spatial_dim=kwargs.get("spatial_dim", 1),
        )

    return net


def get_ae(**model_cfg):
    arch = model_cfg.pop("arch")
    x_dim = model_cfg.pop("x_dim")
    z_dim = model_cfg.pop("z_dim")
    enc_cfg = model_cfg.pop("encoder")
    dec_cfg = model_cfg.pop("decoder")

    if arch == "ae":
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **enc_cfg)
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **dec_cfg)
        ae = AE(encoder, decoder)
    elif arch == "dae":
        sig = model_cfg["sig"]
        noise_type = model_cfg["noise_type"]
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **enc_cfg)
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **dec_cfg)
        ae = DAE(encoder, decoder, sig=sig, noise_type=noise_type)
    elif arch == "wae":
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **enc_cfg)
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **dec_cfg)
        ae = WAE(encoder, decoder, **model_cfg)
    elif arch == "vae":
        sigma_trainable = model_cfg.get("sigma_trainable", False)
        encoder = get_net(in_dim=x_dim, out_dim=z_dim * 2, **enc_cfg)
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **dec_cfg)
        ae = VAE(encoder, decoder, **model_cfg)
    return ae


def get_vae(**model_cfg):
    x_dim = model_cfg["x_dim"]
    z_dim = model_cfg["z_dim"]
    if model_cfg["arch"] == "vae_cent":
        encoder_out_dim = z_dim
    else:
        encoder_out_dim = z_dim * 2

    encoder = get_net(in_dim=x_dim, out_dim=encoder_out_dim, **model_cfg["encoder"])
    decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
    n_sample = model_cfg.get("n_sample", 1)
    pred_method = model_cfg.get("pred_method", "recon")

    if model_cfg["arch"] == "vae":
        ae = VAE(encoder, decoder, n_sample=n_sample, pred_method=pred_method)
    elif model_cfg["arch"] == "vae_sr":
        denoiser = get_net(in_dim=z_dim, out_dim=z_dim, **model_cfg["denoiser"])
        ae = VAE_SR(encoder, decoder, denoiser)
    elif model_cfg["arch"] == "vae_flow":
        flow_cfg = model_cfg["flow"]
        flow = get_glow(**flow_cfg)
        n_kl_sample = model_cfg.get("n_kl_sample", 1)
        ae = VAE_FLOW(
            encoder, decoder, flow, n_sample=n_sample, n_kl_sample=n_kl_sample
        )
    elif model_cfg["arch"] == "vae_proj":
        sample_proj = model_cfg.get("sample_proj", False)
        ae = VAE_PROJ(encoder, decoder, n_sample=n_sample, sample_proj=sample_proj)
    elif model_cfg["arch"] == "vae_cent":
        sig = model_cfg.get("sig", None)
        ae = VAE_ConstEnt(encoder, decoder, n_sample=n_sample, sig=sig)
    return ae


def get_contrastive(**kwargs):
    model_cfg = copy.deepcopy(kwargs["model"])
    x_dim = model_cfg["x_dim"]
    if model_cfg["arch"] == "contrastive_multi":
        x_dim += 1

    net = get_net(in_dim=x_dim, out_dim=1, kwargs=model_cfg["net"])

    if model_cfg["arch"] == "contrastive":
        sigma = model_cfg["sigma"]
        uniform_jitter = model_cfg.get("uniform_jitter", False)
        model = Contrastive(net, sigma=sigma, uniform_jitter=uniform_jitter)
    elif model_cfg["arch"] == "contrastive_v2":
        sigma_1 = model_cfg["sigma_1"]
        sigma_2 = model_cfg["sigma_2"]
        model = ContrastiveV2(net, sigma_1=sigma_1, sigma_2=sigma_2)
    elif model_cfg["arch"] == "contrastive_multi":
        l_sigma = model_cfg["l_sigma"]
        sigma_0 = model_cfg["sigma_0"]
        model = ContrastiveMulti(net, l_sigma=l_sigma, sigma_0=sigma_0)
    return model


def get_glow(**model_cfg):
    if "arch" in model_cfg:
        model_cfg.pop("arch")
    x_dim = model_cfg.pop("x_dim")
    x_size = model_cfg.pop("x_size")
    glow = GlowV2(image_shape=[x_size, x_size, x_dim], **model_cfg)
    return glow


def get_glow_y0ast(**model_cfg):
    if "arch" in model_cfg:
        model_cfg.pop("arch")
    x_dim = model_cfg.pop("x_dim")
    x_size = model_cfg.pop("x_size")
    glow = Glow_y0ast(image_shape=[x_size, x_size, x_dim], **model_cfg)
    return glow


def get_bnaf(**kwargs):
    model_cfg = copy.deepcopy(kwargs["model"])
    model_cfg.pop("arch")
    in_dim = model_cfg["in_dim"]
    model_cfg.pop("in_dim")
    bnaf = BNAF_uniform(data_dim=in_dim, **model_cfg)
    return bnaf


def get_ebm(**model_cfg):
    model_cfg = copy.deepcopy(model_cfg)
    if "arch" in model_cfg:
        model_cfg.pop("arch")
    in_dim = model_cfg["x_dim"]
    model_cfg.pop("x_dim")
    net = get_net(in_dim=in_dim, out_dim=1, **model_cfg["net"])
    model_cfg.pop("net")
    return EnergyBasedModel(net, **model_cfg)


def get_gatedpixelcnn(**model_cfg):
    if "arch" in model_cfg:
        model_cfg.pop("arch")
    model = GatedPixelCNN(n_classes=0, **model_cfg)
    return model


def get_nae(**model_cfg):
    arch = model_cfg.pop("arch")
    x_dim = model_cfg["x_dim"]
    z_dim = model_cfg["z_dim"]

    encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
    decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])

    if arch == "nae":
        ae = NAE(encoder, decoder, **model_cfg["nae"])
    else:
        raise ValueError(f"{arch}")
    return ae


def get_nae_cl(**model_cfg):
    arch = model_cfg.pop("arch")
    sampling = model_cfg.pop("sampling")
    x_dim = model_cfg["x_dim"]
    z_dim = model_cfg["z_dim"]

    encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
    decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
    if arch == "nae_cl" and sampling == "omi":
        sampler_z = get_sampler(**model_cfg["sampler_z"])
        sampler_x = get_sampler(**model_cfg["sampler_x"])
        spatial_dim = model_cfg["decoder"].get("spatial_dim", 1)
        varnet = get_net(
            in_dim=z_dim * spatial_dim * spatial_dim, out_dim=1, **model_cfg["varnet"]
        )
        nae = NAE_CL_OMI(
            encoder, decoder, varnet, sampler_z, sampler_x, **model_cfg["nae"]
        )
    elif arch == "nae_clx" and sampling == "omi":
        sampler_z = get_sampler(**model_cfg["sampler_z"])
        sampler_x = get_sampler(**model_cfg["sampler_x"])
        varnet = get_net(in_dim=x_dim, out_dim=1, **model_cfg["varnet"])
        nae = NAE_CLX_OMI(
            encoder, decoder, varnet, sampler_z, sampler_x, **model_cfg["nae"]
        )
    elif arch == "nae_clzx" and sampling == "omi":
        sampler_z = get_sampler(**model_cfg["sampler_z"])
        sampler_x = get_sampler(**model_cfg["sampler_x"])
        varnetz = get_net(in_dim=z_dim, out_dim=1, **model_cfg["varnetz"])
        varnetx = get_net(in_dim=x_dim, out_dim=1, **model_cfg["varnetx"])
        nae = NAE_CLZX_OMI(
            encoder, decoder, varnetz, varnetx, sampler_z, sampler_x, **model_cfg["nae"]
        )

    elif arch == "nae_l2" and sampling == "omi":
        sampler_z = get_sampler(**model_cfg["sampler_z"])
        sampler_x = get_sampler(**model_cfg["sampler_x"])
        nae = NAE_L2_OMI(encoder, decoder, sampler_z, sampler_x, **model_cfg["nae"])
    else:
        raise ValueError(f"Invalid sampling: {sampling}")
    return nae


def get_pixelcnn(**model_cfg):
    if "arch" in model_cfg:
        model_cfg.pop("arch")

    x_dim = model_cfg["x_dim"]
    nr_resnet = model_cfg["nr_resnet"]
    nr_filters = model_cfg["nr_filters"]
    do_rescale = model_cfg["do_rescale"]
    model = PixelCNN_OC(n_chans=x_dim, nr_resnet=nr_resnet, nr_filters=nr_filters)
    return model


def get_classifier(**model_cfg):
    if "arch" in model_cfg:
        model_cfg.pop("arch")
    x_dim = model_cfg["x_dim"]
    n_class = model_cfg["n_class"]

    return get_net(in_dim=x_dim, out_dim=n_class, **model_cfg["net"])


def get_jem(**model_cfg):
    arch = model_cfg.pop("arch")
    x_dim = model_cfg["x_dim"]
    n_classes = model_cfg["n_classes"]

    net = get_net(in_dim=x_dim, out_dim=n_classes, **model_cfg["net"])
    jem = JEM(net, n_classes=n_classes, **model_cfg["jem"])
    return jem


def get_odin(**model_cfg):
    x_dim = model_cfg["x_dim"]
    n_classes = model_cfg["n_classes"]

    net = get_net(in_dim=x_dim, out_dim=n_classes, **model_cfg["net"])
    odin = ODIN(net)
    return odin


def get_MD(**model_cfg):
    x_dim = model_cfg["x_dim"]
    n_classes = model_cfg["n_classes"]
    net = get_net(in_dim=x_dim, out_dim=n_classes, **model_cfg["net"])
    net_type = model_cfg["net_type"]
    md = MD(net, net_type)
    return md


def get_likelihoodratio(**model_cfg):
    x_dim = model_cfg["x_dim"]
    model = LikelihoodRatio(x_dim=x_dim, pixelcnn_params=model_cfg["pixelcnn_params"])
    return model


def get_likelihoodratio_fore(**model_cfg):
    x_dim = model_cfg["x_dim"]
    model = LikelihoodRatio_fore(
        x_dim=x_dim, pixelcnn_params=model_cfg["pixelcnn_params"]
    )
    return model


def get_likelihoodratio_back(**model_cfg):
    x_dim = model_cfg["x_dim"]
    model = LikelihoodRatio_back(
        x_dim=x_dim, pixelcnn_params=model_cfg["pixelcnn_params"]
    )
    return model


def get_likelihoodratio_v2(fore_model, back_model):
    model = LikelihoodRatio_v2(fore_model, back_model)
    return model


def get_likelihoodregret(**model_cfg):
    x_dim = model_cfg["x_dim"]
    z_dim = model_cfg["z_dim"]
    K = model_cfg["K"]

    sigma_trainable = model_cfg.get("sigma_trainable", False)
    encoder = get_net(
        in_dim=x_dim, out_dim=z_dim * 2, **model_cfg["vae_params"]["encoder"]
    )
    decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["vae_params"]["decoder"])
    b_model = VAE_regret(encoder, decoder, sigma_trainable=sigma_trainable)

    model = LikelihoodRegret(b_model, K=K)
    return model


def get_ic_base_model(**model_cfg):
    """Load pretrained VAE trained on cifar10, reference: https://github.com/XavierXiao/Likelihood-Regret"""
    nz = 100
    nc = 3
    ngf = 64
    ngpu = 1
    image_size = 32
    netG = DCGAN_G(image_size, nz, nc, ngf, ngpu)
    netE = DVAE_E(image_size, nz, nc, ngf, ngpu)

    return netG, netE


def get_stylegan2_g(**model_cfg):
    from models.StyleGAN2.stylegan2 import Generator

    model_cfg.pop("arch")
    generator = Generator(**model_cfg)
    return generator


def get_projgan_g(**model_cfg):
    model_cfg.pop("arch")
    net_type = model_cfg.pop("net")
    cfg = model_cfg

    if net_type == "fastgan":
        from models.PG.pg_modules.networks_fastgan import Generator

        net = Generator(**cfg)
        return net

    elif net_type == "stylegan2":
        from models.PG.pg_modules.networks_stylegan2 import Generator

        net = Generator(**cfg)
        return net
    else:
        raise NotImplementedError


def get_model(cfg, *args, version=None, **kwargs):
    # cfg can be a whole config dictionary or a value of a key 'model' in the config dictionary (cfg['model']).
    if "model" in cfg:
        model_dict = cfg["model"]
    elif "arch" in cfg:
        model_dict = cfg
    else:
        raise ValueError(f"Invalid model configuration dictionary: {cfg}")
    name = model_dict["arch"]
    model = _get_model_instance(name)
    model = model(**model_dict, **kwargs)
    return model


def _get_model_instance(name):
    try:
        return {
            "ae": get_ae,
            "em_ae_v1": get_ae,
            "dae": get_ae,
            "vae": get_ae,
            "glow": get_glow,
            "vae_proj": get_vae,
            "gatedpixelcnn": get_gatedpixelcnn,
            "wae": get_ae,
            "pixelcnn": get_pixelcnn,
            "classifier": get_classifier,
            "md": get_MD,
            "likelihood_ratio": get_likelihoodratio,
            "likelihood_ratio_fore": get_likelihoodratio_fore,
            "likelihood_ratio_back": get_likelihoodratio_back,
            "likelihood_regret": get_likelihoodregret,
            "nae": get_nae,
            "nae_l2": get_nae_cl,
            "glow_y0ast": get_glow_y0ast,
            "resnetcls": ResNetClassifier,
            "stylegan2_g": get_stylegan2_g,
            "projgan_g": get_projgan_g,
        }[name]
    except:
        raise ValueError("Model {} not available".format(name))


def load_pretrained(identifier, config_file, ckpt_file, root="pretrained", **kwargs):
    """
    load pre-trained model.
    identifier: '<model name>/<run name>'. e.g. 'ae_mnist/z16'
    config_file: name of a config file. e.g. 'ae.yml'
    ckpt_file: name of a model checkpoint file. e.g. 'model_best.pth'
    root: path to pretrained directory
    """
    config_path = os.path.join(root, identifier, config_file)
    ckpt_path = os.path.join(root, identifier, ckpt_file)
    cfg = OmegaConf.load(config_path)
    model_name = cfg["model"]["arch"]

    if model_name == "md":
        data_path = kwargs.get("data_path", "datasets")
        device = kwargs.get("device", "cpu")
        lr_tunned_with = kwargs.get("lr_tunned_with", "SVHN_OOD")
        return load_pretrained_md(
            cfg,
            ckpt_path,
            data_path,
            root=root,
            device=device,
            lr_tunned_with=lr_tunned_with,
        )
    elif model_name == "glow":
        model = get_model(cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model_state" in ckpt:
            ckpt = ckpt["model_state"]
        model.load_state_dict(ckpt)
        model.eval()
        model.set_actnorm_init(inited=True)
    elif model_name == "csi":
        return load_pretrained_csi(cfg, root, identifier, **kwargs)
    elif model_name == "msp":
        return load_pretrained_oe(cfg, root, identifier, ckpt_file, **kwargs)
    elif model_name == "oe_tune":
        return load_pretrained_oe(cfg, root, identifier, ckpt_file, **kwargs)
    elif model_name == "oe_scratch":
        return load_pretrained_oe(cfg, root, identifier, ckpt_file, **kwargs)
    elif model_name == "ssd":
        return load_pretrained_ssd(cfg, root, identifier, ckpt_file, **kwargs)
    elif model_name == "good":
        return load_pretrained_good(cfg, root, identifier, ckpt_file, **kwargs)
    elif model_name == "ensemble":
        device = kwargs.get("device", "cpu")
        return load_pretrained_ensemble(cfg, device)
    elif model_name == "atom":
        return load_pretrained_atom(cfg, root, identifier, ckpt_file, method="atom")
    elif model_name == "rowl":
        return load_pretrained_atom(cfg, root, identifier, ckpt_file, method="rowl")
    elif model_name == "due":
        return load_pretrained_due(cfg, root, identifier, ckpt_file)
    elif model_name == "sngp":
        return load_pretrained_due(cfg, root, identifier, ckpt_file)
    elif model_name == "vit_tf":
        return load_pretrained_vit_tf(cfg, root, identifier, ckpt_file)
    elif model_name == "vit_hf":
        return load_pretrained_vit_hf(cfg, root, identifier, ckpt_file)
    elif model_name == "prood":
        return load_pretrained_prood(cfg, root, identifier, ckpt_file)
    elif model_name == "plain_from_prood" or model_name == "oe_from_prood":
        return load_pretraiend_plain_or_oe_from_prood(cfg, root, identifier, ckpt_file)
    else:
        model = get_model(cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model_state" in ckpt:
            ckpt = ckpt["model_state"]
        elif "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt)
        model.eval()
    return model, cfg


def load_pretrained_good(cfg, root, identifier, ckpt_file, **kwargs):
    from models.GOOD.provable_classifiers import CNN_IBP

    model = CNN_IBP(dset_in_name="CIFAR10", size="XL")
    path = os.path.join(root, identifier, ckpt_file)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model, cfg


def load_pretrained_ssd(cfg, root, identifier, ckpt_file, device="cuda:0", **kwargs):
    from models.SSD.models_SSD import SSLResNet

    model = SSLResNet(arch="resnet50").eval()
    model.encoder = torch.nn.DataParallel(
        model.encoder,
        device_ids=[torch.device(device)],
        output_device=torch.device(device),
    )
    ckpt = os.path.join(root, identifier, ckpt_file)

    # load checkpoint
    ckpt_dict = torch.load(ckpt, map_location="cpu")
    if "model" in ckpt_dict.keys():
        ckpt_dict = ckpt_dict["model"]
    if "state_dict" in ckpt_dict.keys():
        ckpt_dict = ckpt_dict["state_dict"]
    model.load_state_dict(ckpt_dict)

    model.initialize_ftrain(
        torch.load(os.path.join(root, identifier, "ftrain.pt"), map_location="cpu")
    )
    return model, cfg


def load_pretrained_csi(cfg, root, identifier, **kwargs):
    class Arg:
        def __init__(self):
            pass

    P = Arg()
    P.mode = "ood_pre"
    P.dataset = "cifar10"
    P.model = "resnet18"
    P.ood_score = "CSI"
    P.shift_trans_type = "rotation"
    P.load_path = "cifar10_unlabeled"
    P.ood_dataset = "svhn"
    P.local_rank = 0
    P.resize_factor = 0.54
    P.resize_fix = True
    P.batch_size = 32
    P.ood_samples = 10
    P.ood_layer = ["simclr", "shift"]
    P.ood_eval = True
    image_size = (3, 32, 32)
    P.image_size = image_size
    P.n_classes = 10

    simclr_aug = C.get_simclr_augmentation(P, image_size=P.image_size)
    P.shift_trans, P.K_shift = C.get_shift_module(P, eval=True)
    P.shift_trans = P.shift_trans

    model = C.get_classifier(P.model, n_classes=P.n_classes)
    model = C.get_shift_classifer(model, P.K_shift)

    checkpoint = torch.load(
        os.path.join(root, identifier, kwargs["model_path"]),
        map_location=kwargs.get("device", "cpu"),
    )
    model.load_state_dict(checkpoint, strict=not True)

    hflip = TL.HorizontalFlipLayer()

    feats_train = {}
    feats_train["shift"] = torch.load(
        os.path.join(root, identifier, kwargs["shift_path"])
    )
    feats_train["simclr"] = torch.load(
        os.path.join(root, identifier, kwargs["simclr_path"])
    )

    P.axis = []
    for f in feats_train["simclr"].chunk(P.K_shift, dim=1):
        axis = f.mean(dim=1)  # (M, d)
        P.axis.append(normalize(axis, dim=1))

    f_sim = [
        f.mean(dim=1) for f in feats_train["simclr"].chunk(P.K_shift, dim=1)
    ]  # list of (M, d)
    f_shi = [
        f.mean(dim=1) for f in feats_train["shift"].chunk(P.K_shift, dim=1)
    ]  # list of (M, 4)

    weight_sim = []
    weight_shi = []
    for shi in range(P.K_shift):
        sim_norm = f_sim[shi].norm(dim=1)  # (M)
        shi_mean = f_shi[shi][:, shi]  # (M)
        weight_sim.append(1 / sim_norm.mean().item())
        weight_shi.append(1 / shi_mean.mean().item())

    P.weight_sim = weight_sim
    P.weight_shi = weight_shi

    kwargs = {
        "simclr_aug": simclr_aug,
        "sample_num": P.ood_samples,
        "layers": P.ood_layer,
    }

    from models.CSI.csi import CSI_detector

    return CSI_detector(P, model, **kwargs), cfg


def load_pretrained_md(
    cfg,
    ckpt_path,
    data_path,
    root="pretrained",
    lr_tunned_with="SVHN_OOD",
    device="cpu",
):
    """Load pretrained Mahalanobis Distance detector (Lee et al., 2017) model"""
    model = get_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state" in ckpt.keys():
        model.load_state_dict(ckpt["model_state"])
    else:
        model.net.load_state_dict(ckpt)

    if cfg["data"]["training"]["dataset"] == "CIFAR10_OOD":
        path = os.path.join(root, "cifar_ood_md/md_resnet/")
        # augmentations = {
        #     'normalize': {
        #         'mean': (0.4914, 0.4822, 0.4465),
        #         'std': (0.2023, 0.1994, 0.2010)
        #         },
        # } is used when finding 'lr' but not for classification training
    elif cfg["data"]["training"]["dataset"] == "FashionMNISTpad_OOD":
        path = os.path.join(root, "fmnist32_ood_md/md_resnet/")

    samples_mean = torch.load(
        os.path.join(path, "samples_mean.pt"), map_location=device
    )
    precision = torch.load(os.path.join(path, "precision.pt"), map_location=device)
    num_output = torch.load(os.path.join(path, "num_output.pt"), map_location=device)
    num_classes = torch.load(os.path.join(path, "num_classes.pt"), map_location=device)
    model.sample_mean = samples_mean
    model.precision = precision
    model.num_output = num_output
    model.num_classes = num_classes

    if lr_tunned_with == "SVHN_OOD":
        lr = torch.load(
            os.path.join(path, "model_lr_tunned_with_SVHN_OOD.pt"), map_location=device
        )
    elif lr_tunned_with == "MNISTpad_OOD":
        lr = torch.load(
            os.path.join(path, "model_lr_tunned_with_MNIST_OOD.pt"), map_location=device
        )
    else:
        pass

    model.lr = lr
    return model, cfg


def load_pretrained_lr(
    config_path_fore, ckpt_path_fore, config_path_back, ckpt_path_back, device="cpu"
):
    """load pretrained likelihood ratio for outlier detection (Ren et al., 2019) model"""
    cfg_fore = OmegaConf.load(config_path_fore)
    fore_model = get_model(cfg_fore)
    fore_model.load_state_dict(
        torch.load(ckpt_path_fore, map_location=device)["model_state"]
    )

    cfg_back = OmegaConf.load(config_path_back)
    back_model = get_model(cfg_back)
    back_model.load_state_dict(
        torch.load(ckpt_path_back, map_location=device)["model_state"]
    )

    cfg = {"cfg_fore": cfg_fore, "cfg_back": cfg_back}
    model = LikelihoodRatio_v2(fore_model, back_model)
    return model, cfg


def load_pretrained_oe(cfg, root, identifier, ckpt_file, **kwargs):
    if kwargs["network"] == "allconv":
        net = AllConvNet(kwargs["num_classes"])
    elif kwargs["network"] == "wrn":
        widen_factor = 2
        droprate = 0.3
        net = WideResNet(40, kwargs["num_classes"], widen_factor, droprate)

    ckpt_path = os.path.join(root, identifier, ckpt_file)
    net.load_state_dict(torch.load(ckpt_path))
    net.eval()
    model = OutlierExposure(net)
    return model, cfg


def load_pretrained_ensemble(cfg, device):
    l_detector = []
    l_no_grad = []
    train_dl = get_dataloader(cfg["data"]["indist_val"])
    for key, cfg_detector in cfg["detector"].items():
        model, _ = load_pretrained(**cfg_detector, device=device)

        if "detector_aug" in cfg_detector:
            aug = get_composed_augmentations(cfg_detector["detector_aug"])
        else:
            aug = None
        no_grad = cfg_detector.get("detector_no_grad", False)
        l_no_grad.append(no_grad)
        model = Detector(model, bound=-1, transform=aug, no_grad=no_grad, use_rank=True)
        model.to(device)

        identifier = cfg_detector["identifier"]
        if identifier == "cifar_ood_good":
            in_score = (
                model.learn_normalization(dataloader=train_dl, device=device)
                .detach()
                .cpu()
                .numpy()
            )
        else:
            normalization_path = os.path.join(
                "pretrained", identifier, f"rank_normalization.pkl"
            )
            if os.path.isfile(normalization_path):
                model.load_normalization(normalization_path, device)
            else:
                in_score = (
                    model.learn_normalization(dataloader=train_dl, device=device)
                    .detach()
                    .cpu()
                    .numpy()
                )
                writer.add_histogram("in_train_score", in_score, 0)
                if not do_ensemble:
                    model.save_normalization(normalization_path)
        l_detector.append(model)
    detector = EnsembleDetector(l_detector, bound=-1, use_rank=True)
    detector.to(device)
    return detector, cfg


# def load_stylegan2_generator(cfg, root, identifier, ckpt_file, **kwargs):
#     from models.StyleGAN2.stylegan2 import Generator
#
#     ckpt = torch.load(os.path.join(root, identifier, ckpt_file))["state_dict"]
#     cfg = cfg["model"]
#     cfg.pop("arch")
#     generator = Generator(**cfg)
#     generator.load_state_dict(ckpt)
#     generator.eval()
#     return generator, cfg


def load_pretrained_atom(cfg, root, identifier, ckpt_file, **kwargs):
    import models.ATOM.models.densenet as dn
    from models.ATOM.atom import ATOM, ROWL

    if cfg["in_dataset"] == "CIFAR-10":
        normalizer = transforms.Normalize(
            (125.3 / 255, 123.0 / 255, 113.9 / 255),
            (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0),
        )
        num_classes = 10
        num_reject_classes = 1
        layers = 100
    method = cfg["method"]

    net = dn.DenseNet3(layers, num_classes + num_reject_classes, normalizer=normalizer)
    checkpoint = os.path.join(root, identifier, ckpt_file)
    net.load_state_dict(torch.load(checkpoint)["state_dict"])
    net.eval()
    if method == "atom":
        model = ATOM(net, num_classes=num_classes)
    elif method == "rowl":
        model = ROWL(net, num_classes=num_classes)
    return model, cfg


def load_pretrained_due(cfg, root, identifier, ckpt_file, **kwargs):
    from gpytorch.likelihoods import SoftmaxLikelihood

    from models.DUE.due import DUE, SNGP
    from models.DUE.duelib import dkl
    from models.DUE.duelib.wide_resnet import WideResNet

    method = cfg["model"]["arch"]
    checkpoint_path = os.path.join(root, identifier, ckpt_file)

    input_size = 32
    spectral_conv = True
    spectral_bn = True
    dropout_rate = 0.3
    coeff = 3
    n_power_iterations = 1

    feature_extractor = WideResNet(
        input_size,
        spectral_conv,
        spectral_bn,
        dropout_rate=dropout_rate,
        coeff=coeff,
        n_power_iterations=n_power_iterations,
    )

    if method == "due":
        n_inducing_points = 10
        kernel = "RBF"
        # no need to informative initialization -- we will load if from the checkpoint anyway
        initial_inducing_points = torch.randn(10, 640, dtype=torch.float)
        initial_lengthscale = 0.1
        num_classes = 10

        # initial_inducing_points, initial_lengthscale = dkl.initial_values(
        #     train_dataset, feature_extractor, n_inducing_points
        # )

        gp = dkl.GP(
            num_outputs=num_classes,
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            kernel=kernel,
        )

        net = dkl.DKL(feature_extractor, gp)

        net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        likelihood = SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)

        model = DUE(net, likelihood)
    elif method == "sngp":
        from models.DUE.duelib.sngp import Laplace

        num_deep_features = 640
        num_gp_features = 128
        normalize_gp_features = True
        num_random_features = 1024
        mean_field_factor = 25
        ridge_penalty = 1
        feature_scale = 2

        # hard coded for CIFAR-10
        num_data = 50000
        num_classes = 10
        batch_size = 128

        net = Laplace(
            feature_extractor,
            num_deep_features,
            num_gp_features,
            normalize_gp_features,
            num_random_features,
            num_classes,
            num_data,
            batch_size,
            ridge_penalty,
            feature_scale,
            mean_field_factor,
        )
        net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model = SNGP(net)

    return model, cfg


def load_pretrained_prood(cfg, root, identifier, ckpt_file):
    from models.prood import Prood
    from models.prood.models import provable_classifiers, resnet

    cfg = cfg["model"]
    cfg_detector = cfg["detector"]
    cfg_classifier = cfg["classifier"]
    detector = provable_classifiers.CNN_IBP(
        dset_in_name=cfg_detector.dset_in_name,
        size=cfg_detector.arch_size,
        last_bias=cfg_detector.use_last_bias,
        num_classes=cfg_detector.num_classes,
        last_layer_neg=cfg_detector.last_layer_neg,
    )

    if cfg_detector.last_layer_neg and cfg_detector.use_last_bias:
        with torch.no_grad():
            detector.layers[-1].bias.data += cfg_detector.bias_shift

    classifier = resnet.get_ResNet(dset=cfg_classifier.dset_in_name)
    model = Prood(classifier, detector, cfg_classifier.num_classes)

    checkpoint = os.path.join(root, identifier, ckpt_file)
    model.net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    return model, cfg

def load_pretraiend_plain_or_oe_from_prood(cfg, root, identifier, ckpt_file):
    from models.prood import Plain_or_OE_from_Prood
    from models.prood.models.resnet import get_ResNet
    arch_dset_in_name = cfg["model"]["classifier"].dset_in_name
    classifier = get_ResNet(dset=arch_dset_in_name)
    
    model = Plain_or_OE_from_Prood(classifier)
    
    checkpoint = os.path.join(root, identifier, ckpt_file)
    model.net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    return model, cfg
    