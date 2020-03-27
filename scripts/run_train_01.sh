#!/bin/bash



## change time, gpu, mem, cpus, all name
module load anaconda3/5.0.1
module load cuda/10.0
module load cudnn/v7.4-cuda.10.0
source activate my_env



NET=${14}
rot=$1
DIM=$9
rot_only=${12}
pretrained=${13}
save_step=50
Epoch=600
Model_LIST=`seq  50 50 600`
DATA_ROOT=../Dataset
Gallery_eq_Query=True
R=.pth.tar
LR=0.00001
BatchSize=80
RATIO=0.16
BETA=2



DATA=$2
LOSS=$3
ROT_LR=$4
ALPHA=$5
rot_bt=$6
rot_batch=$7
LR=$8
up_step=${10}
num_clusters=${11}

use_test=0

if [ -z "$rot_only" ]
then
    rot_only=0
fi
if [ -z "$pretrained" ]
then
    pretrained=True
fi 
if [ -z "$NET" ]
then
    NET=Inception
fi













target=results/Deep_Metric





root=/gpfs/data/xl6/xuefei/${target}
CHECKPOINTS=${root}/ckpt









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



SAVE_DIR=${CHECKPOINTS}/${LOSS}/${DATA}/${NET}-DIM-${DIM}-lr-${LR}-ratio-${RATIO}-BatchSize-${BatchSize}-rot_lr-${ROT_LR}-self_supervision_rot-${rot}-rot_bt-${rot_bt}-rot_batch-${rot_batch}-up_step-${up_step}-num_clusters-${num_clusters}-ALPHA-${ALPHA}-BETA-${BETA}

if_exist_mkdir ${SAVE_DIR}
width=227




num_instances=5


echo "Begin Training!"
stdbuf -oL python ../train.py --net ${NET} \
--data $DATA \
--self_supervision_rot $rot \
--rot_bt $rot_bt \
--rot_batch $rot_batch \
-g_eq_q ${Gallery_eq_Query} \
--pretrained ${pretrained} \
--pool_feature ${POOL_FEATURE:-'False'} \
--use_test $use_test \
--loss $LOSS \
--rot_only $rot_only \
--data_root ${DATA_ROOT} \
--init random \
--resume yes \
--lr $LR \
--rot_lr $ROT_LR \
--dim $DIM \
--alpha $ALPHA \
--beta $BETA \
--num_instances ${num_instances} \
--batch_size ${BatchSize} \
--num_clusters ${num_clusters} \
--epoch ${Epoch} \
--width ${width} \
--save_dir ${SAVE_DIR} \
--save_step ${save_step} \
--up_step ${up_step} \
--ratio ${RATIO} 

echo "Begin Testing!"

save_txt=${root}/result/$LOSS/$DATA/${NET}-DIM-${DIM}-lr-${LR}-ratio-${RATIO}-BatchSize-${BatchSize}-rot_lr-${ROT_LR}-self_supervision_rot-${rot}-rot_bt-${rot_bt}-rot_batch-${rot_batch}-up_step-${up_step}-num_clusters-${num_clusters}-ALPHA-${ALPHA}-BETA-${BETA}.txt

for i in $Model_LIST; do
    CUDA_VISIBLE_DEVICES=0 python ../test.py --net ${NET} \
    --data $DATA \
    --net $NET \
    --dim $DIM \
    --save_txt ${save_txt} \
    --data_root ${DATA_ROOT} \
    --batch_size 100 \
    -g_eq_q ${Gallery_eq_Query} \
    --width ${width} \
    --resume ${SAVE_DIR}/ckp_ep$i$R \
    --pool_feature ${POOL_FEATURE:-'False'} \
    | tee -a ${save_txt}
done





