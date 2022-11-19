#! /bin/bash

# evaluation of adversarial samples
# pairwise
device=1
dataset=svhn
# dataset=cars

in_dataset=CIFAR10
in_dataset_lower=cifar
# in_dataset=RImgNet
# in_dataset_lower=rimgnet

# variation=coord
# variation=colorV2_mh
# variation=colorV1
# variation=stylegan2ada_mh
# variation=stylegan2ada_z512_mh
# mode=pairwise
mode=pairwise

echo $dataset

if [ $mode = single ]
then

# for variation in affineV1_random colorV1_random
for variation in pgstylegan2_z16_mh 
do
    echo $variation
    # for model in due  
    for model in ae pixelcnn nae good acet ceda ssd md oe csi sngp atom rowl csi prood vit_hf_md
    # for model in glow
    # for model in nae good acet ceda ssd md oe csi sngp atom rowl csi prood vit_hf_md
    # for model in sngp atom rowl
    do
        target=${model}
        echo $model;
        # python evaluate_advdist_single.py \
        #     --resultdir results_attack/${dataset}_${variation}/cifar_${model}_${dataset}_${variation}/run \
        #     --config cifar_${model}_${dataset}_${variation}.yml \
        #     --device ${device}
        python evaluate_advdist_pairwise.py \
                --detector configs/${in_dataset_lower}_detectors/${in_dataset_lower}_${model}.yml \
                --target results/${in_dataset}/${target}/${dataset}_${variation}/run/${in_dataset_lower}_${target}_${dataset}_${variation}.yml\
                --logdir results/${in_dataset}/pairwise/${model}/${target}/${dataset}_${variation}\
                --device ${device} \
                --in-dataset ${in_dataset} 
    done
done

else

# for variation in affineV2_mh colorV1_mh pgstylegan2_z16_mh; do
for variation in  pgstylegan2_z16_mh; do
# for model in vit_hf_md 
# for model in msp oe prood vit_hf_md
for model in  nae good acet ceda ssd md oe csi sngp atom rowl csi prood vit_hf_md
# for model in ae pixelcnn glow nae good acet ceda ssd md oe csi 
# for model in nae good acet ceda ssd md oe csi 
do
    # for target in nae good acet ceda ssd md atom oe rowl csi 
    # for target in vit_hf_md
    # for target in msp oe prood vit_hf_md
    for target in  nae good acet ceda ssd md oe csi sngp atom rowl csi prood vit_hf_md
    do
    # target=${model}

    cmd="python evaluate_advdist_pairwise.py \
        --detector configs/${in_dataset_lower}_detectors/${in_dataset_lower}_${model}.yml \
        --target results/${in_dataset}/${target}/${dataset}_${variation}/run/${in_dataset_lower}_${target}_${dataset}_${variation}.yml\
        --logdir results/${in_dataset}/pairwise/${model}/${target}/${dataset}_${variation}\
        --device auto \
        --in-dataset ${in_dataset}" 
    echo $cmd
    sleep 0.1s
    tsp ${cmd}

    done
done
done


fi
