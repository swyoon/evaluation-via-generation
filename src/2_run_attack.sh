#! /bin/bash

CFG_ATT=configs/cifar_attacks
CFG_DET=configs/cifar_detectors
# DATASET=celeba
# THREAT=affine
# THREAT=stylegan2_mh
# for threat in coord affine colorV1 stylegan2_mh;
# for threat in stylegan2ada_z16_mh stylegan2ada_z32_mh stylegan2ada_z512_mh;
for DATASET in svhn celeba ;
do
# for threat in stylegan2ada_z16_lgv stylegan2ada_z32_mh stylegan2ada_z64_mh stylegan2ada_z512_mh;
# for threat in stylegan2ada_z16_lgv stylegan2ada_z32_lgv stylegan2ada_z64_lgv stylegan2ada_z512_lgv;
# for threat in stylegan2ada_z16_mh;
for threat in affineV1_mh colorV2_mh;
do
    THREAT=${threat}
    split=1

    for model in acet ae ceda csi glow good md nae oe pixelcnn ssd prood vit_hf_md;
    # for model in oe;
    # for model in prood vit_hf_md ;
    do
        for ((idx=0;idx<split;idx++)); do
        echo $model
        # tsp python attack.py --attack ${CFG_ATT}/${DATASET}_${THREAT}.yml \
        cmd="python attack.py --attack ${CFG_ATT}/${DATASET}_${THREAT}.yml \
            --detector ${CFG_DET}/cifar_${model}.yml \
            --logdir results/ \
            --run run \
            --n_sample 1000 --split ${split} --idx ${idx} \
            --device auto  \
            --data.out_eval.batch_size 250" 
            # --advdist.trurncation_psi 0.1
            # --device $((idx))  \
        echo ${cmd}
        tsp ${cmd}

        sleep 0.1s
        done
    done
done
done
