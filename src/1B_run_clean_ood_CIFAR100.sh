#! /bin/bash

# Reproduce OOD detection performance of pretrained models
device=3
# ood=SVHN_OOD,CelebA_OOD
ood=SVHN_OOD
configs=(cifar100_vit.yml
     )

for config in "${configs[@]}"; do
   echo ${config}
   python evaluate_ood.py --dataset CIFAR100_OOD --ood ${ood} \
        --config configs/cifar100_detectors/${config} --device ${device} --split test 
done
