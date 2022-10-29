#! /bin/bash

CFG_ATT=configs/cifar_attacks
CFG_DET=configs/cifar_detectors
# DATASET=celeba
# THREAT=affine
# THREAT=stylegan2_mh
# for threat in coord affine colorV1 stylegan2_mh;
# for threat in stylegan2ada_z16_mh stylegan2ada_z32_mh stylegan2ada_z512_mh;
for DATASET in svhn ;
do
# for threat in stylegan2ada_z16_lgv stylegan2ada_z32_mh stylegan2ada_z64_mh stylegan2ada_z512_mh;
for threat in stylegan2ada_z16_lgv stylegan2ada_z32_lgv stylegan2ada_z64_lgv stylegan2ada_z512_lgv;
# for threat in stylegan2ada_z16_lgv;
do
    THREAT=${threat}
    split=2

    # for model in acet ae ceda csi glow good md nae oe pixelcnn ssd;
    for model in oe;
    # for model in vit_hf_md ;
    do
        for ((idx=0;idx<split;idx++)); do
        echo $model
        # tsp python attack.py --attack ${CFG_ATT}/${DATASET}_${THREAT}.yml \
        echo "python attack.py --attack ${CFG_ATT}/${DATASET}_${THREAT}.yml \
            --detector ${CFG_DET}/cifar_${model}.yml \
            --logdir results/ \
            --run run \
            --n_sample 100 --split ${split} --idx ${idx} \
            --device auto  \
            --data.out_eval.batch_size 50"
            # --device $((idx))  \

        sleep 0.1s
        done
    done
done
done
