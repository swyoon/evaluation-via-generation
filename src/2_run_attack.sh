#! /bin/bash

CFG_ATT=configs/cifar_attacks
CFG_DET=configs/cifar_detectors
DATASET=svhn
# DATASET=svhn
# THREAT=affine
# THREAT=stylegan2_mh
# for threat in coord affine colorV1 stylegan2_mh;
for threat in affine ;
do
    THREAT=${threat}
    split=1

    # for model in acet ae ceda csi glow good md nae oe pixelcnn ssd;
    for model in oe ;
    do
        for ((idx=0;idx<split;idx++)); do
        echo $model
        # tsp python attack.py --attack ${CFG_ATT}/${DATASET}_${THREAT}.yml \
        python attack.py --attack ${CFG_ATT}/${DATASET}_${THREAT}.yml \
            --detector ${CFG_DET}/cifar_${model}.yml \
            --logdir results/cifar_${model}/${DATASET}_${THREAT}  \
            --run run \
            --n_sample 100 --split ${split} --idx ${idx} \
            --device $((idx))  \
            --data.out_eval.batch_size 100
        done
    done
done
