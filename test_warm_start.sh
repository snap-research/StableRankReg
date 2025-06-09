#!/bin/bash


DATASETS=('MovieLens1M' 'Gowalla' 'Yelp2018' 'AmazonBook')


echo $DATASET
SEEDS=(123 246 492)
LOSSES=('BPR' 'align')
HIDDEN_DIMS=(64)

for DATASET in "${DATASETS[@]}" ; do
    for SEED in "${SEEDS[@]}" ; do
        for LOSS in "${LOSSES[@]}" ; do

            if [ $LOSS = "BPR"  ]; then
                REGS=("-1")
            elif [ $LOSS = "align"  ]; then
                REGS=("uniformity")
            fi

            for REG in  "${REGS[@]}" ; do

                for HIDDEN_DIM in "${HIDDEN_DIMS[@]}" ; do
                    python test_warm_start.py --dataset $DATASET --model MLP --seed $SEED --loss $LOSS --hidden_dim $HIDDEN_DIM --reg_types $REG --model_save_path "./models_chkp_warmstart"
                    python test_warm_start.py --dataset $DATASET --model LGConv --seed $SEED --loss $LOSS --hidden_dim $HIDDEN_DIM --reg_types $REG --model_save_path "./models_chkp_warmstart"
                done
            done
        done
    done
done
