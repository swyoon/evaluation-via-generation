#! /bin/bash

# evaluation of adversarial samples
# pairwise
device=1
# dataset=svhn
dataset=cars
# dataset=fgvc
# dataset=flowers
in_dataset=RImgNet
in_dataset_lower=rimgnet
# variation=coord
variation=affine
# variation=colorV1
# variation=stylegan2ada_mh
# variation=stylegan2ada_z512_mh
mode=pairwise
# mode=pairwise
# postfix=v0115

echo $dataset

if [ $mode = single ]
then

for variation in coord affine colorV1 stylegan2_mh
# for variation in stylegan2_mh
do
    echo $variation
    for model in due  
    # for model in ae pixelcnn glow nae good acet ceda ssd md oe csi 
    do
        echo $model;
        python evaluate_advdist_single.py \
            --resultdir results_attack/${dataset}_${variation}/cifar_${model}_${dataset}_${variation}/run \
            --config cifar_${model}_${dataset}_${variation}.yml \
            --device ${device}
    done
done

else

# for model in vit_hf_md 
for model in prood 
# for model in ae pixelcnn glow nae good acet ceda ssd md oe csi 
# for model in nae good acet ceda ssd md oe csi 
do
    # for target in nae good acet ceda ssd md atom oe rowl csi 
    # for target in vit_hf_md
    for target in prood 
    do
    # target=${model}

    python evaluate_advdist_pairwise.py \
        --detector configs/${in_dataset_lower}_detectors/${in_dataset_lower}_${model}.yml \
        --target results/${in_dataset}/${target}/${dataset}_${variation}/run/${in_dataset_lower}_${target}_${dataset}_${variation}.yml\
        --logdir results/${in_dataset}/pairwise/${model}/${target}/${dataset}_${variation}\
        --device ${device} \
        --in-dataset ${in_dataset} \

    done
done


fi
