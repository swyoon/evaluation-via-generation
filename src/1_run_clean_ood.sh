#! /bin/bash

# Reproduce OOD detection performance of pretrained models
device=3
ood=SVHN_OOD,CelebA_OOD
# ood=""
configs=(# cifar_ae.yml
         # cifar_pixelcnn.yml 
         # cifar_md.yml
         # cifar_oe.yml
         # cifar_nae.yml
         cifar_glow.yml
         cifar_csi.yml
         cifar_ssd.yml
         cifar_good.yml
         cifar_acet.yml
         cifar_ceda.yml
         cifar_rowl.yml
         cifar_atom.yml
         cifar_due.yml
         cifar_sngp.yml
         cifar_prood.yml
     )
configs=(cifar_prood.yml cifar_vit_hf_md.yml)
# configs=(cifar_prood.yml)
# configs=()

for config in "${configs[@]}"; do
   echo ${config}
    python evaluate_ood.py --dataset CIFAR10_OOD --ood ${ood} \
        --config configs/cifar_detectors/${config} --device ${device} --in_split evaluation 

    # tsp python evaluate_ood.py --dataset CIFAR10_OOD  \
    #     --config configs/cifar_detectors/${config} --device auto --in_split training_full 
    # sleep 0.1s
done




# ood=Cars,Flowers,FGVC
# ood=dtd
# ood=OpenImages-O,EuroSAT
# ood=FGVC,Flowers,EuroSAT
ood=EuroSAT
configs=(rimgnet_vit_hf_md.yml)
# configs=( rimgnet_vit_hf_md.yml rimgnet_prood.yml )
# configs=( rimgnet_plain.yml rimgnet_oe.yml )
configs=()
for config in "${configs[@]}"; do
   echo ${config}
    python evaluate_ood.py --dataset RImgNet --ood ${ood} \
        --config configs/rimgnet_detectors/${config} --device ${device} --in_split evaluation 
done


