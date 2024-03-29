#! /bin/bash

CFG_ATT=configs/cifar_attacks
CFG_DET=configs/cifar_detectors
# DATASET=celeba
# THREAT=affine
# THREAT=stylegan2_mh
# for DATASET in svhn celeba ;
for DATASET in  svhn ;
do
# for threat in stylegan2ada_z16_lgv stylegan2ada_z32_mh stylegan2ada_z64_mh stylegan2ada_z512_mh;
# for threat in stylegan2ada_z16_lgv stylegan2ada_z32_lgv stylegan2ada_z64_lgv stylegan2ada_z512_lgv;
# for threat in stylegan2ada_z16_mh;
# for threat in affineV1_mh colorV2_mh;
# for threat in affineV1_random colorV1_random;
# for threat in colorV1_mh ;
for threat in affineV0_mh  ;
do
    THREAT=${threat}
    split=4

    # for model in acet ae ceda csi  good md nae atom sngp rowl oe pixelcnn ssd prood vit_hf_md;
    # for model in oe ssd prood vit_hf_md;
    for model in acet ceda good sngp atom;
    # for model in sngp atom rowl;
    # for model in prood vit_hf_md ;
    # for model in vit_hf_md ;
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

        sleep 0.4s
        done
    done
done
done
