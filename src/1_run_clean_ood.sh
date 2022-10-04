#! /bin/bash

# Reproduce OOD detection performance of pretrained models
device=3
ood=SVHN_OOD,CelebA_OOD
# ood=CIFAR100_OOD
# ood=CelebA_OOD
# models=(ae pixcelcnn++ oe md)
models=(ae
        pixelcnn++ 
        md
        oe
        nae
        glow
        csi
        SSD
        GOOD
        ACET
        CEDA
        ROWL
        ATOM
        DUE
        SNGP)

for model in "${models[@]}"; do
    aug=None
    custom_args=""

    if [[ $model == "ae" ]]; then
        model_dir=pretrained/cifar_ood_ae/ghosh_z128
        ckpt=model_best.pkl
        config=ghosh_z128.yml
    elif [[ $model == "pixelcnn++" ]]; then
        model_dir=pretrained/cifar_ood_pixelcnn/f80/ 
        ckpt=model_best.pkl
        config=f80.yml
    elif [[ $model == "md" ]]; then
        model_dir=pretrained/cifar_ood_md/md_resnet
        ckpt=resnet_cifar10.pth
        config=md_resnet_cifar.yml
        aug=CIFAR10
        custom_args="--lr_tunned_with SVHN_OOD"
    elif [[ $model == "nae" ]]; then
        model_dir=pretrained/cifar_ood_nae/z32gn/
        ckpt=nae_8.pkl
        config=z32gn.yml
    elif [[ $model == "glow" ]]; then
        model_dir=pretrained/cifar_ood_glow/logit_deq
        ckpt=model_best.pkl
        config=glow.yml
    elif [[ $model == "csi" ]]; then
        model_dir=pretrained/cifar_ood_csi/
        ckpt=null
        config=csi.yml
        custom_args="--model_path cifar10_unlabeled.model --shift_path feats_10_resize_fix_0.54_cifar10_train_shift.pth --simclr_path feats_10_resize_fix_0.54_cifar10_train_simclr.pth"
    elif [[ $model == "oe" ]]; then
        model_dir=pretrained/cifar_ood_oe_scratch/allconv/ 
        ckpt=cifar10_allconv_oe_scratch_epoch_99.pt
        config=oe_scratch_allconv.yml
        aug=CIFAR10
        custom_args="--method outlier_exposure"
    elif [[ $model == "SSD" ]]; then
        model_dir=pretrained/cifar_ood_ssd
        ckpt=model_best.pth.tar
        config=ssd.yml
        aug=CIFAR10
    elif [[ $model == "GOOD" ]]; then
        model_dir=pretrained/cifar_ood_good/good80
        ckpt=GOODQ80.pt
        config=good.yml
    elif [[ $model == "ACET" ]]; then
        model_dir=pretrained/cifar_ood_good/acet
        ckpt=ACET.pt
        config=good.yml
    elif [[ $model == "CEDA" ]]; then
        model_dir=pretrained/cifar_ood_good/ceda
        ckpt=CEDA.pt
        config=good.yml
    elif [[ $model == "ROWL" ]]; then
        model_dir=pretrained/cifar_ood_atom/rowl
        ckpt=checkpoint_100.pth.tar
        config=rowl.yml
    elif [[ $model == "ATOM" ]]; then
        model_dir=pretrained/cifar_ood_atom/atom
        ckpt=checkpoint_100.pth.tar
        config=atom.yml
    elif [[ $model == "DUE" ]]; then
        model_dir=pretrained/cifar_ood_due/due
        ckpt=due.pt
        config=due.yml
        aug=CIFAR10
    elif [[ $model == "SNGP" ]]; then
        model_dir=pretrained/cifar_ood_due/sngp
        ckpt=sngp.pt
        config=sngp.yml
        aug=CIFAR10
    fi

   echo ${model} ${model_dir}
    python evaluate_ood.py --dataset CIFAR10_OOD --ood ${ood} --resultdir ${model_dir} \
        --ckpt ${ckpt} --config ${config} --device ${device} --split test \
        --aug "${aug}" ${custom_args}
done

# #  AE
# python evaluate_ood.py --dataset CIFAR10_OOD --ood ${ood} --resultdir pretrained/cifar_ood_ae/ghosh_z128/ --ckpt model_best.pkl --config ghosh_z128.yml \
#                          --device ${device}
# 
# ## PixelCNN++
# python evaluate_ood.py --dataset CIFAR10_OOD --ood ${ood} --resultdir pretrained/cifar_ood_pixelcnn/f80/ --ckpt model_best.pkl --config f80.yml \
#                          --device ${device}
# 
# # MD
# python evaluate_ood.py --dataset CIFAR10_OOD --ood ${ood} --resultdir pretrained/cifar_ood_md/md_resnet/ --ckpt resnet_cifar10.pth --config md_resnet_cifar.yml \
#                          --device ${device} --aug CIFAR10 --lr_tunned_with SVHN_OOD
# 
# ## NAE
# python evaluate_ood.py --dataset CIFAR10_OOD --ood ${ood} --resultdir pretrained/cifar_ood_nae/z32gn/ --ckpt nae_8.pkl --config z32gn.yml --device ${device}
# 
# ## GLOW
# python evaluate_ood.py --dataset CIFAR10_OOD --ood ${ood} --resultdir pretrained/cifar_ood_glow/logit_deq/ --ckpt model_best.pkl --config glow.yml --device ${device}
# 
# ## CSI
# python evaluate_ood.py --dataset CIFAR10_OOD --ood ${ood} --resultdir pretrained/cifar_ood_csi/ --ckpt null --config csi.yml --device ${device} \
#                        --model_path cifar10_unlabeled.model --shift_path feats_10_resize_fix_0.54_cifar10_train_shift.pth --simclr_path feats_10_resize_fix_0.54_cifar10_train_simclr.pth
# 
# ## Outlier Exposure
# python evaluate_ood.py --dataset CIFAR10_OOD --ood ${ood} --resultdir pretrained/cifar_ood_oe_scratch/allconv/ --ckpt cifar10_allconv_oe_scratch_epoch_99.pt --config oe_scratch_allconv.yml \
#                        --device ${device} --aug CIFAR10 --method outlier_exposure
# 
# ### Ensemble
# # python evaluate_ood.py --dataset CIFAR10_OOD --ood ${ood} --resultdir pretrained/cifar_ood_ensemble --ckpt model_best.pkl --config csi_md_nae_oe.yml \
# #                         --device ${device}
# 
# SSD
#  python evaluate_ood.py --dataset CIFAR10_OOD --ood ${ood} --resultdir pretrained/cifar_ood_ssd --ckpt model_best.pth.tar --config ssd.yml \
#                           --device ${device} --aug CIFAR10
#  
#  
#  ## GOOD80
#  python evaluate_ood.py --dataset CIFAR10_OOD --ood ${ood} --resultdir pretrained/cifar_ood_good/good80 --ckpt GOODQ80.pt --config good.yml \
#                           --device ${device}
#  
#  
#  ## ACET
#  python evaluate_ood.py --dataset CIFAR10_OOD --ood ${ood} --resultdir pretrained/cifar_ood_good/acet --ckpt ACET.pt --config good.yml \
#                           --device ${device}
#  
#  ## CEDA
#  python evaluate_ood.py --dataset CIFAR10_OOD --ood ${ood} --resultdir pretrained/cifar_ood_good/ceda --ckpt CEDA.pt --config good.yml \
#                           --device ${device}
#  
#  
#  ## ROWL
#  python evaluate_ood.py --dataset CIFAR10_OOD --ood ${ood} --resultdir pretrained/cifar_ood_atom/rowl --ckpt checkpoint_100.pth.tar --config rowl.yml \
#                           --device ${device}
#  
#  ## ATOM
#  python evaluate_ood.py --dataset CIFAR10_OOD --ood ${ood} --resultdir pretrained/cifar_ood_atom/atom --ckpt checkpoint_100.pth.tar --config atom.yml \
#                           --device ${device}
#  
#  ## DUE 
#  python evaluate_ood.py --dataset CIFAR10_OOD --ood ${ood} --resultdir pretrained/cifar_ood_due/due --ckpt due.pt --config due.yml \
#                           --device ${device} --aug CIFAR10
#  
#  ## SNGP 
#  python evaluate_ood.py --dataset CIFAR10_OOD --ood ${ood} --resultdir pretrained/cifar_ood_due/sngp --ckpt sngp.pt --config sngp.yml \
#                           --device ${device} --aug CIFAR10
