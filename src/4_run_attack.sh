#! /bin/bash

# Reproduce OOD detection performance of pretrained models
device=2
# ood=SVHN_OOD,CelebA_OOD
ood=""
configs=(
        # cifar_ae.yml
        #  cifar_pixelcnn.yml 
        #  cifar_md.yml
         # cifar_oe.yml
        #  cifar_nae.yml
        #  cifar_glow.yml
        #  cifar_csi.yml
        #  cifar_ssd.yml
        #  cifar_good.yml
        #  cifar_acet.yml
        #  cifar_ceda.yml
        #  cifar_rowl.yml
        #  cifar_atom.yml
        #  cifar_due.yml
        #  cifar_sngp.yml
        #  cifar_prood.yml
         # rimgnet_prood.yml
         rimgnet_vit_hf_md.yml
     )

for config in "${configs[@]}"; do
   echo ${config}
    # python evaluate_ood.py --dataset CIFAR10_OOD --ood ${ood} \
    python evaluate_grad_attack_ood.py --dataset RImgNet  --ood FGVC \
        --config configs/rimgnet_detectors/${config} --device ${device} --n_sample 400 --normalize
done
