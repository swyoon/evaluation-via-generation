#! /bin/bash

CFG_ATT=configs/rimgnet_attacks
CFG_DET=configs/rimgnet_detectors
# DATASET=celeba
# THREAT=affine
# THREAT=stylegan2_mh
# for threat in coord affine colorV1 stylegan2_mh;
# for threat in stylegan2ada_z16_mh stylegan2ada_z32_mh stylegan2ada_z512_mh;
for DATASET in flowers cars fgvc;
do
# for threat in stylegan2ada_z16_lgv stylegan2ada_z32_mh stylegan2ada_z64_mh stylegan2ada_z512_mh;
for threat in affine colorV1;
do
    THREAT=${threat}
    split=4

    # for model in acet ae ceda csi glow good md nae oe pixelcnn ssd;
    # for model in prood;
    for model in vit_hf_md ;
    do
        for ((idx=0;idx<split;idx++)); do
        echo $model
        # tsp python attack.py --attack ${CFG_ATT}/${DATASET}_${THREAT}.yml \
        tsp python attack.py --attack ${CFG_ATT}/${DATASET}_${THREAT}.yml \
            --detector ${CFG_DET}/rimgnet_${model}.yml \
            --logdir results/ \
            --run run \
            --n_sample 400 --split ${split} --idx ${idx} \
            --device auto  \
            --data.out_eval.batch_size 5
            # --device $((idx))  \

        sleep 0.1s
        done
    done
done
done
