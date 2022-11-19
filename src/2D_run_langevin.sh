#! /bin/bash

# CFG_ATT=configs/rimgnet_attacks
# CFG_DET=configs/rimgnet_detectors
CFG_ATT=configs/cifar_attacks
CFG_DET=configs/cifar_detectors
# DATASET=celeba
# THREAT=affine
# THREAT=stylegan2_mh
for DATASET in svhn ;
do
# for threat in affineV0_random affineV0_mh affineV0_lgv;
for threat in pgstylegan2_z64_randomwalk pgstylegan2_z64_mh pgstylegan2_z64_lgv pgstylegan2_z64_random ;
do
    THREAT=${threat}
    split=1

    # for model in acet ae ceda csi glow good md nae oe pixelcnn ssd;
    # for model in oe;
    # for noise_std in 0.01 0.05 0.1 0.2;
    for noise_std in 0.01;
    do
    for repeat in 0 1 2 3 4;
    do
    for model in oe prood vit_hf_md ;
    do
        for ((idx=0;idx<split;idx++)); do
        echo $model
        # tsp python attack.py --attack ${CFG_ATT}/${DATASET}_${THREAT}.yml \
        cmd="python attack.py --attack ${CFG_ATT}/${DATASET}_${THREAT}.yml \
            --detector ${CFG_DET}/cifar_${model}.yml \
            --logdir results/ \
            --run benchmark_${repeat} \
            --n_sample 100 --split ${split} --idx ${idx} \
            --device auto  \
            --data.out_eval.batch_size 100 \
            --save_intermediate"
            # --advdist.sampler.stepsize ${noise_std} \
            # --advdist.sampler.noise_std ${noise_std}"
            # --device $((idx))  \
        echo ${cmd}
        tsp ${cmd}

        sleep 0.3s
        done
    done
    done
    done
done
done
