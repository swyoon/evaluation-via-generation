#! /bin/bash

CFG_ATT=configs/cifar_attacks
CFG_DET=configs/cifar_detectors
# DATASET=svhn
DATASET=celeba
# THREAT=affine
# THREAT=stylegan2_mh
# for threat in coord affine colorV1 stylegan2_mh;
for threat in stylegan2ada_z16_mh stylegan2ada_z32_mh stylegan2ada_z512_mh;
do
    THREAT=${threat}
    split=4

    # for model in acet ae ceda csi glow good md nae oe pixelcnn ssd;
    for model in prood ;
    do
        for ((idx=0;idx<split;idx++)); do
        echo $model
        # tsp python attack.py --attack ${CFG_ATT}/${DATASET}_${THREAT}.yml \
        tsp python attack.py --attack ${CFG_ATT}/${DATASET}_${THREAT}.yml \
            --detector ${CFG_DET}/cifar_${model}.yml \
            --logdir results/CIFAR10/${model}/${DATASET}_${THREAT}  \
            --run run \
            --n_sample 5000 --split ${split} --idx ${idx} \
            --device auto  \
            --data.out_eval.batch_size 250
            # --device $((idx))  \

        sleep 0.1s
        done
    done
done
