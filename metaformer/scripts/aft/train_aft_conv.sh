DATA_PATH="D:\Code\Paper-code\metaformer\cifar-100"
CODE_PATH="D:\Code\Paper-code\metaformer"

ALL_BATCH_SIZE=256
NUM_GPU=1
GRAD_ACCUM_STEPS=1  # 根据你的GPU数量和内存进行调整

let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS

cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model aftconv --opt lamb --lr 8e-3 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path 0.6 --head-dropout 0.5