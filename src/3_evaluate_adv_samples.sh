#! /bin/bash

# evaluation of adversarial samples
# pairwise
device=0
dataset=svhn
# dataset=celeba
# variation=coord
# variation=affine
# variation=colorV1
variation=stylegan2ada_mh
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

for model in atom 
# for model in ae pixelcnn glow nae good acet ceda ssd md oe csi 
# for model in nae good acet ceda ssd md oe csi 
do
    # for target in nae good acet ceda ssd md atom oe rowl csi 
    for target in atom 
    do
    # target=${model}

    python evaluate_advdist_pairwise.py \
        --detector configs/cifar_detectors/cifar_${model}.yml \
        --target results/CIFAR10/${target}/${dataset}_${variation}/run/cifar_${target}_${dataset}_${variation}.yml\
        --logdir results/CIFAR10/pairwise/${dataset}/${variation}\
        --device ${device}

    done
done


fi
