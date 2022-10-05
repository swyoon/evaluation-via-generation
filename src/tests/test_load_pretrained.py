import pytest

from models import load_pretrained

# (identifier, config_file, ckpt_file, kwargs)
cifar_glow = ("cifar_ood_glow/logit_deq", "glow.yml", "model_best.pkl", {})
cifar_md = (
    "cifar_ood_md/md_resnet",
    "md_resnet_cifar.yml",
    "resnet_cifar10.pth",
    {"lr_tunned_with": "SVHN_OOD"},
)
fmnist_md = (
    "fmnist_ood_md/md_resnet",
    "md_resnet_fmnist.yml",
    "model_best.pkl",
    {"lr_tunned_with": "MNISTpad_OOD"},
)
cifar_ood_nae = ("cifar_ood_nae/z32gn", "z32gn.yml", "nae_8.pkl", {})
cifar_ood_nae_ae = (
    "cifar_ood_nae/z32gn",
    "z32gn.yml",
    "model_best.pkl",
    {},
)  # NAE before training
mnist_ood_nae = ("mnist_ood_nae/z32", "z32.yml", "nae_20.pkl", {})
mnist_ood_nae_ae = (
    "mnist_ood_nae/z32",
    "z32.yml",
    "model_best.pkl",
    {},
)  # NAE before training
celeba64_ood_nae = ("celeba64_ood_nae/z64gr_h32g8", "z64gr_h32g8.yml", "nae_3.pkl", {})


fmnist_vae = ("fmnist_ood_vae/z32lik", "z32lik.yml", "model_epoch_96.pkl", {})

mnist32_ood_ae = ("mnist32_ood_ae/z32", "z32.yml", "model_epoch_300.pkl", {})
mnist32_ood_vqvae = (
    "mnist32_ood_vqvae/K10_4x4",
    "K10_4x4.yml",
    "model_epoch_280.pkl",
    {},
)
cifar_v_svhn_res18 = ("cifar_v_svhn/res18_lr5", "res18.yml", "model_best.pkl", {})
fmnist_v_mnist_res18 = ("fmnist_v_mnist/res18_lr5", "res18.yml", "model_best.pkl", {})
cifar_v_svhnceleba_res50 = (
    "cifar_v_svhnceleba/res50",
    "res50.yml",
    "model_best.pkl",
    {},
)
cifar_v_svhn_res50norm = (
    "cifar_v_svhn/res50norm",
    "res50norm.yml",
    "model_epoch_10.pkl",
    {},
)
cifar_v_cifar100_res50norm = (
    "cifar_v_cifar100/res50norm",
    "res50norm.yml",
    "model_10.pkl",
    {},
)
fmnist32_ood_pixelcnn = ("fmnist32_ood_pixelcnn/f80", "f80.yml", "model_best.pkl", {})
cifar_ood_ssd = ("cifar_ood_ssd", "ssd.yml", "model_best.pth.tar", {})
cifar_ood_good_q = (
    "cifar_ood_good/good80",
    "good.yml",
    "GOODQ80.pt",
    {},
)  # f'GOOD{n}.pt' s.t. n \in {0, 20, 40, 60, 80, 90, 95, 100}
cifar_ood_good_acet = ("cifar_ood_good/acet", "good.yml", "ACET.pt", {})
cifar_ood_good_ceda = ("cifar_ood_good/ceda", "good.yml", "CEDA.pt", {})
cifar_ood_good_oe = ("cifar_ood_good", "good.yml", "OE.pt", {})
cifar_ood_good_plain = ("cifar_ood_good", "good.yml", "Plain.pt", {})

## autoencoders
mnist32fmnist32_ood_vqvae = (
    "mnist32fmnist32_ood_vqvae/K10_4x4",
    "K10_4x4.yml",
    "model_epoch_280.pkl",
    {},
)
mnist32fmnist32_ood_ae = (
    "mnist32fmnist32_ood_ae/z32",
    "z32.yml",
    "model_epoch_300.pkl",
    {},
)
cifarsvhn_ood_ae = ("cifarsvhn_ood_ae/z32", "z32.yml", "model_epoch_300.pkl", {})
cifarsvhn_ood_ae32 = (
    "cifarsvhn_ood_ae/z32nh32",
    "z32nh32.yml",
    "model_epoch_300.pkl",
    {},
)
cifarsvhn_ood_ae64 = ("cifarsvhn_ood_ae/z64", "z64.yml", "model_epoch_500.pkl", {})
cifarsvhn_ood_ae128 = ("cifarsvhn_ood_ae/z128", "z128.yml", "model_best.pkl", {})
cifarsvhn_ood_ae256 = ("cifarsvhn_ood_ae/z256", "z256.yml", "model_best.pkl", {})
cifarsvhn_ood_ae512 = ("cifarsvhn_ood_ae/z512", "z512.yml", "model_best.pkl", {})
cifarsvhnceleba_ood_ae = (
    "cifarsvhnceleba_ood_ae/z32",
    "z32.yml",
    "model_epoch_500.pkl",
    {},
)
cifarsvhnceleba_ood_ae64 = (
    "cifarsvhnceleba_ood_ae/z64",
    "z64.yml",
    "model_epoch_500.pkl",
    {},
)

cifar_csi = (
    "cifar_ood_csi",
    "csi.yml",
    "null",
    {
        "model_path": "cifar10_unlabeled.model",
        "shift_path": "feats_10_resize_fix_0.54_cifar10_train_shift.pth",
        "simclr_path": "feats_10_resize_fix_0.54_cifar10_train_simclr.pth",
    },
)

cifar_oe = (
    "cifar_ood_oe_scratch/allconv",
    "oe_scratch_allconv.yml",
    "cifar10_allconv_oe_scratch_epoch_99.pt",
    {"network": "allconv", "num_classes": 10},
)
cifar_oe_imgnet = (
    "cifar_ood_oe_scratch/allconv",
    "oe_scratch_allconv.yml",
    "cifar10_allconv_oe_scratch_imagenetaug_epoch_149.pt",
    {"network": "allconv", "num_classes": 10},
)

svhn_stylegan2_gen = (
    "svhn_stylegan2/z64",
    "generator.yml",
    "model=G_ema-best-weights-step=188000.pth",
    {},
)
svhn_stylegan2_gen_ada_test_z16 = (
    "svhn_stylegan2/ada_test_z16",
    "generator.yml",
    "model=G_ema-current-weights-step=200000.pth",
    {},
)
svhn_stylegan2_gen_ada_test_z32 = (
    "svhn_stylegan2/ada_test_z32",
    "generator.yml",
    "model=G_ema-current-weights-step=200000.pth",
    {},
)
svhn_stylegan2_gen_ada_test_z64 = (
    "svhn_stylegan2/ada_test_z64",
    "generator.yml",
    "model=G_ema-current-weights-step=200000.pth",
    {},
)
svhn_stylegan2_gen_ada_test_z512 = (
    "svhn_stylegan2/ada_test_z512",
    "generator.yml",
    "model=G_ema-current-weights-step=200000.pth",
    {},
)


celeba32_stylegan2_gen = (
    "celeba32_stylegan2/z64",
    "generator.yml",
    "model=G_ema-best-weights-step=194000.pth",
    {},
)
celeba32_stylegan2_gen_ada_test_z16 = (
    "celeba32_stylegan2/ada_test_z16",
    "generator.yml",
    "model=G_ema-current-weights-step=200000.pth",
    {},
)
celeba32_stylegan2_gen_ada_test_z32 = (
    "celeba32_stylegan2/ada_test_z32",
    "generator.yml",
    "model=G_ema-current-weights-step=200000.pth",
    {},
)
celeba32_stylegan2_gen_ada_test_z64 = (
    "celeba32_stylegan2/ada_test_z64",
    "generator.yml",
    "model=G_ema-current-weights-step=200000.pth",
    {},
)
celeba32_stylegan2_gen_ada_test_z512 = (
    "celeba32_stylegan2/ada_test_z512",
    "generator.yml",
    "model=G_ema-current-weights-step=200000.pth",
    {},
)

cifar_ood_atom = ("cifar_ood_atom/atom", "atom.yml", "checkpoint_100.pth.tar", {})
cifar_ood_rowl = ("cifar_ood_atom/rowl", "rowl.yml", "checkpoint_100.pth.tar", {})

cifar_ood_due = ("cifar_ood_due/due", "due.yml", "due.pt", {})
cifar_ood_sngp = ("cifar_ood_due/sngp", "sngp.yml", "sngp.pt", {})

l_setting = [
    cifar_glow,
    cifar_md,
    fmnist_vae,
    mnist32_ood_ae,
    cifar_ood_nae,
    cifar_v_svhn_res18,
    fmnist_v_mnist_res18,
    cifar_v_svhnceleba_res50,
    fmnist32_ood_pixelcnn,
    mnist32fmnist32_ood_ae,
    cifar_csi,
    cifar_oe,
    cifarsvhn_ood_ae,
    cifarsvhn_ood_ae128,
    cifarsvhn_ood_ae256,
    cifarsvhn_ood_ae512,
    cifarsvhn_ood_ae32,
    cifarsvhnceleba_ood_ae,
    cifarsvhnceleba_ood_ae64,
    cifar_ood_ssd,
    cifar_ood_good_q,
    cifar_ood_good_acet,
    cifar_ood_good_ceda,
    cifar_ood_good_oe,
    cifar_ood_good_plain,
    svhn_stylegan2_gen,
    svhn_stylegan2_gen_ada_test_z16,
    svhn_stylegan2_gen_ada_test_z32,
    svhn_stylegan2_gen_ada_test_z64,
    svhn_stylegan2_gen_ada_test_z512,
    celeba32_stylegan2_gen,
    celeba32_stylegan2_gen_ada_test_z16,
    celeba32_stylegan2_gen_ada_test_z32,
    celeba32_stylegan2_gen_ada_test_z64,
    celeba32_stylegan2_gen_ada_test_z512,
    cifar_ood_atom,
    cifar_ood_rowl,
    cifar_ood_due,
    cifar_ood_sngp,
]


@pytest.mark.parametrize("model_setting", l_setting)
def test_load_pretrained(model_setting):
    identifier, config_file, ckpt_file, kwargs = model_setting
    model, cfg = load_pretrained(identifier, config_file, ckpt_file, **kwargs)
