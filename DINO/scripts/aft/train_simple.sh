DATA_PATH="/root/autodl-tmp/metaformer/cifar-100"
CODE_PATH="/root/autodl-tmp/metaformer"
MODEL_PATH="/root/autodl-tmp/metaformer/output/train/20240725-185023-aftfull-224/model_best.pth.tar"

ALL_BATCH_SIZE=64
NUM_GPU=1
GRAD_ACCUM_STEPS=1  # 根据你的GPU数量和内存进行调整

let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS

cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model aftfull --opt lamb --lr 8e-3 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--initial-checkpoint $MODEL_PATH \
--drop-path 0.6 --head-dropout 0.5
