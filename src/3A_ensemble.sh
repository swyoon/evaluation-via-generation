#! /bin/bash

# evaluation of adversarial samples
# pairwise
device=2
in_dataset=CIFAR10
in_dataset_lower=cifar
# in_dataset=RImgNet
# in_dataset_lower=rimgnet

# in_dataset_lower=rimgnet
# variation=coord
# variation=colorV2_mh
# variation=colorV1
# variation=stylegan2ada_mh
# variation=stylegan2ada_z512_mh
# mode=pairwise

echo $dataset


# for variation in svhn_affineV1 
# for variation in svhn_colorV1 celeba_colorV1 celeba_affineV1
# for variation in fgvc_colorV1 fgvc_affineV2 fgvc_pgstylegan2_z16 flowers_colorV1 flowers_affineV2 flowers_pgstylegan2_z16 eurosat_affineV2 eurosat_colorV1 eurosat_pgstylegan2_z16
for variation in svhn_colorV1 svhn_affineV1 svhn_pgstylegan2_z16 celeba_colorV1 celeba_affineV1 celeba_pgstylegan2_z16
do
    echo $variation
    # for model in due  
    for model in ae pixelcnn glow nae good acet ceda ssd md oe csi sngp atom rowl csi prood vit_hf_md
    # for model in msp oe vit_hf_md prood
    # for model in glow
    # for model in nae good acet ceda ssd md oe csi sngp atom rowl csi prood vit_hf_md
    # for model in sngp atom rowl
    do
        target=${model}
        echo $model;
        python ensemble_attack.py \
                --detector configs/${in_dataset_lower}_detectors/${in_dataset_lower}_${model}.yml \
                --outlier ${variation} \
                --ensemble V1 \
                --logdir results/ \
                --device ${device} 
    done
done

