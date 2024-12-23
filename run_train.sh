set -e

source ~/.bashrc

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))
}

if [ ! $GPU_NUM ]; then
GPU_NUM=1
fi

export MODEL_TYPE=EAI
export LR_ONE_EPOCH=0.001
export BATCHSZIE_ONE_EPOCH=64
export DATAPATH=./Dataset_GRAB

echo "GPU_NUM : " $GPU_NUM
echo "MODEL_TYPE : " $MODEL_TYPE
echo "LR : " $LR_ONE_EPOCH
echo "BATCHSZIE_ONE_EPOCH : " $BATCHSZIE_ONE_EPOCH
echo "DATAPATH : " $DATAPATH

rnd=$[$(rand 1 1000)+4231]
echo "RANDOM_PORT : " $rnd

EXPID=TRAIN_modeltype_${MODEL_TYPE}_batchsize_$[$BATCHSZIE_ONE_EPOCH]_lr_$LR_ONE_EPOCH
TESTEXPID=TEST_modeltype_${MODEL_TYPE}_batchsize_$[$BATCHSZIE_ONE_EPOCH]_lr_$LR_ONE_EPOCH
echo "EXPID : " $EXPID

python -u -m torch.distributed.run \
--nproc_per_node=$GPU_NUM \
--master_port=$rnd \
train.py \
--input_n 30 \
--output 30 \
--all_n 60 \
--lr $LR_ONE_EPOCH \
--train_batch $[$BATCHSZIE_ONE_EPOCH*$GPU_NUM] \
--model_type $MODEL_TYPE \
--grab_data_dict $DATAPATH \
--is_exp \
--is_using_saved_file \
--is_using_noTpose2 \
--is_boneloss \
--exp $EXPID \




