#!/bin/bash


DATASET='MovieLens1M'
MODEL='MLP'
EPOCHS=500
LOSS='BPR'
echo $DATASET

BATCH_SIZE=8192
SEEDS=(123)
HIDDEN_DIMS=(64)
NUM_LAYERS=(0) 
WEIGHT_DECAYS=(0.0001 0.000001 0.00000001)
LRS=(0.1 0.01 0.001)

if [ $LOSS = "BPR"  ]; then
    BATCH_SIZE=16384
    NEG_RATIOS=(1)
elif [ $LOSS = "align"  ]; then
    NEG_RATIOS=(0)
    BATCH_SIZE=8192
fi


for SEED in "${SEEDS[@]}" ; do
    for HIDDEN_DIM in "${HIDDEN_DIMS[@]}" ; do
        for NUM_LAYER in "${NUM_LAYERS[@]}" ; do
            for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}" ; do
                for LR in "${LRS[@]}" ; do
                    for NEG_RATIO in "${NEG_RATIOS[@]}" ; do
                        python train.py --model $MODEL --dataset $DATASET --epochs $EPOCHS \
                                        --seed $SEED --loss $LOSS --hidden_dim $HIDDEN_DIM --num_layers $NUM_LAYER --weight_decay $WEIGHT_DECAY \
                                        --lr $LR --batch_size $BATCH_SIZE --neg_ratio $NEG_RATIO --overwrite True --reg_types "-1" \
                                        --gamma_vals "-1" --warm_start_epochs 5 --warm_start_reg_types "stable_rank" --warm_start_gamma_vals "1.0"
                        wait

                    done
                done
            done
        done
    done
done

