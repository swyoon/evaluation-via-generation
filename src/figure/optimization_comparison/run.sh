#! /bin/bash


# for method in lgv mh random randomwalk grad; do
for method in grad; do
    for model in oe prood vit_hf_md; do
        for manifold in affineV0 pgstylegan2_z64; do
            for idx in 0 1 2 3 4; do
                cmd="python benchmark_additional_langevin.py --method ${method} --model ${model} --manifold ${manifold} --idx ${idx} --device auto"
                echo ${cmd}
                tsp ${cmd}
            done
        done
    done
done
