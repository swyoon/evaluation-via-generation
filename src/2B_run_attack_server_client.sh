#! /bin/bash

CFG_ATT=configs/cifar_attacks
CFG_DET=configs/cifar100_detectors
DATASET=svhn
# DATASET=celeba
# THREAT=affine
# THREAT=stylegan2_mh
# for threat in coord affine colorV1 stylegan2_mh;
# for threat in stylegan2ada_z16_mh stylegan2ada_z32_mh stylegan2ada_z64_mh stylegan2ada_z512_mh;
for threat in stylegan2ada_z16_mh; 
do
    THREAT=${threat}
    split=4
    idx=$1

    # for model in acet ae ceda csi glow good md nae oe pixelcnn ssd;
    for model in vit ;
    do
        echo $model
        CUDA_VISIBLE_DEVICES=$idx DETECTOR=configs/cifar_detectors/cifar_atom.yml uvicorn launch_detector_server:app --port 33333  >> server.log&

        CUDA_VISIBLE_DEVICES=$idx python attack.py --attack ${CFG_ATT}/${DATASET}_${THREAT}.yml \
            --detector ${CFG_DET}/cifar100_${model}.yml \
            --logdir results \
            --run run \
            --n_sample 1000 --split ${split} --idx ${idx} \
            --device 0  \
            --data.out_eval.batch_size 250
            # --device $((idx))  \

        kill `cat server.${idx}.pid`

        sleep 0.1s
    done
done
