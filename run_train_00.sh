#!/bin/bash
#SBATCH --time=2000
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --partition=learnfair
#SBATCH --mem 400G
#SBATCH --cpus-per-task 80


## change time, gpu, mem, cpus, all name
module load anaconda3/5.0.1
module load cuda/10.0
module load cudnn/v7.4-cuda.10.0
source activate my_env


DATA_ROOT=data
Gallery_eq_Query=True
root=/checkpoint/xuefeicao01/metric_learning/Deep_Metric
CHECKPOINTS=${root}/ckpt
NET=BN-Inception
DIM=512
R=.pth.tar
LR=0.00001
BatchSize=80
RATIO=0.16
BETA=2



rot=$1
DATA=$2
LOSS=$3
ROT_LR=$4
ALPHA=$5
rot_bt=$6

declare -rot rot
declare -DATA DATA
declare -LOSS LOSS
declare -ROT_LR ROT_LR
declare -ALPHA ALPHA
declare -rot_bt rot_bt

echo $rot
echo $DATA
echo $LOSS
echo $ROT_LR
echo $ALPHA
echo $rot_bt




if_exist_mkdir ()
{
    dirname=$1
    if [ ! -d "$dirname" ]; then
    mkdir $dirname
    fi
}

if_exist_mkdir ${root}

if_exist_mkdir ${CHECKPOINTS}
if_exist_mkdir ${CHECKPOINTS}/${LOSS}
if_exist_mkdir ${CHECKPOINTS}/${LOSS}/${DATA}

if_exist_mkdir ${root}/result
if_exist_mkdir ${root}/result/${LOSS}
if_exist_mkdir ${root}/result/${LOSS}/${DATA}

save_step=50
Epoch=600
Model_LIST=`seq  50 50 600`

SAVE_DIR=${CHECKPOINTS}/${LOSS}/${DATA}/${NET}-DIM-${DIM}-lr-${LR}-ratio-${RATIO}-BatchSize-${BatchSize}-rot_lr-${ROT_LR}-self_supervision_rot-${rot}-rot_bt-${rot_bt}-ALPHA-${ALPHA}-BETA-${BETA}
if_exist_mkdir ${SAVE_DIR}

if [ "$DATA" = product ]
then
  echo "batch size increase to 1000"
  BatchSize=1000
  save_step=25
  Epoch=400
  Model_LIST=`seq  25 25 400`
fi 

if [ "$DATA" = car ]
then
  echo "epoch size increase to 2000"
  Epoch=2000
  Model_LIST=`seq  50 50 2000`
fi 




# if [ ! -n "$1" ] ;then
echo "Begin Training!"
python train.py --net ${NET} \
--data $DATA \
--self_supervision_rot $rot \
--rot_bt $rot_bt \
--loss $LOSS \
--data_root ${DATA_ROOT} \
--init random \
--resume yes \
--lr $LR \
--rot_lr $ROT_LR \
--dim $DIM \
--alpha $ALPHA \
--beta $BETA \
--num_instances 5 \
--batch_size ${BatchSize} \
--epoch ${Epoch} \
--width 227 \
--save_dir ${SAVE_DIR} \
--save_step ${save_step} \
--ratio ${RATIO} 

echo "Begin Testing!"


for i in $Model_LIST; do
    CUDA_VISIBLE_DEVICES=0 python test.py --net ${NET} \
    --data $DATA \
    --data_root ${DATA_ROOT} \
    --batch_size 100 \
    -g_eq_q ${Gallery_eq_Query} \
    --width 227 \
    -r ${SAVE_DIR}/ckp_ep$i$R \
    --pool_feature ${POOL_FEATURE:-'False'} \
    | tee -a ${root}/result/$LOSS/$DATA/${NET}-DIM-${DIM}-ratio-${RATIO}-Batchsize-${BatchSize}-rot_lr-${ROT_LR}-self_supervision_rot-${rot}-rot_bt-${rot_bt}-ALPHA-${ALPHA}-BETA-${BETA}-lr-$LR${POOL_FEATURE:+'-pool_feature'}.txt

done


