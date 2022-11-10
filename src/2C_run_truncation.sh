#! /bin/bash

CFG_ATT=configs/rimgnet_attacks
CFG_DET=configs/rimgnet_detectors
# DATASET=celeba
# THREAT=affine
# THREAT=stylegan2_mh
# for threat in coord affine colorV1 stylegan2_mh;
# for threat in stylegan2ada_z16_mh stylegan2ada_z32_mh stylegan2ada_z512_mh;
# for DATASET in celeba ;
for DATASET in eurosat ;
do
# for threat in stylegan2ada_z16_lgv stylegan2ada_z32_mh stylegan2ada_z64_mh stylegan2ada_z512_mh;
# for threat in stylegan2ada_z16_lgv stylegan2ada_z32_lgv stylegan2ada_z64_lgv stylegan2ada_z512_lgv;
for threat in pgstylegan2_z16_mh;
do
    THREAT=${threat}
    split=1

    # for model in acet ae ceda csi glow good md nae oe pixelcnn ssd;
    # for model in oe;
    for psi in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9;
    do
    for model in vit_hf_md ;
    do
        for ((idx=0;idx<split;idx++)); do
        echo $model
        # tsp python attack.py --attack ${CFG_ATT}/${DATASET}_${THREAT}.yml \
        cmd="python attack.py --attack ${CFG_ATT}/${DATASET}_${THREAT}.yml \
            --detector ${CFG_DET}/rimgnet_${model}.yml \
            --logdir results/ \
            --run psi_${psi} \
            --n_sample 50 --split ${split} --idx ${idx} \
            --device auto  \
            --data.out_eval.batch_size 250 \
            --advdist.truncation_psi ${psi}"
            # --device $((idx))  \
        echo ${cmd}
        tsp ${cmd}

        sleep 0.1s
        done
    done
    done
done
done
