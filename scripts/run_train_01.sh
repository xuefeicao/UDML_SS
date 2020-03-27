#!/bin/bash
#SBATCH --time=4000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=priority
#SBATCH --mem 120G
#SBATCH --cpus-per-task 10
#SBATCH --constraint=volta32gb
#SBATCH --comment="cvpr suppl deadline Nov 22"
#SBATCH --mail-user=xuefeicao01@fb.com
#SBATCH --mail-type=end # mail once the job finishes


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
DATA_ROOT=../data
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




if [ "${num_clusters}" = 500 ] || [ "${num_clusters}" = 1000 ] || [ "${num_clusters}" = 2000 ]
then
  echo "increase batch size"
  BatchSize=240
  if [ "$DATA" = cub ]
  then
    BatchSize=80
  fi 
  if [ "$DATA" = car ]
  then
    BatchSize=80
  fi 
fi 
if [ "${num_clusters}" = 3000 ]
then
  echo "increase batch size"
  BatchSize=240
  save_step=25
  Epoch=400
  Model_LIST=`seq  50 50 400`
  if [ "$DATA" = cub ]
  then
    BatchSize=80
  fi 
  if [ "$DATA" = car ]
  then
    BatchSize=80
  fi 
fi
if [ "${num_clusters}" = 4000 ] || [ "${num_clusters}" = 5000 ] || [ "${num_clusters}" = 6000 ] || [ "${num_clusters}" = 7000 ] || [ "${num_clusters}" = 8000 ] || [ "${num_clusters}" = 9000 ] || [ "${num_clusters}" = 10000 ]
then
  echo "increase batch size"
  BatchSize=240
  save_step=25
  Epoch=300
  Model_LIST=`seq  25 25 300`
  if [ "$DATA" = cub ]
  then
    BatchSize=80
  fi 
  if [ "$DATA" = car ]
  then
    BatchSize=80
  fi 
fi
if [ "$DATA" = shop ]
then
  echo "gallery is not equal to query"
  Gallery_eq_Query=False
  save_step=10
  Epoch=400
  Model_LIST=`seq  10 10 400`
fi
if [ "$DATA" = cifar ]
then
  echo "gallery is not equal to query"
  Gallery_eq_Query=False
fi  
if [ "$DATA" = car ]
then
  echo "epoch size increase to 2000"
  save_step=100
  Epoch=2000
  Model_LIST=`seq  100 100 2000`
fi 

if [ "$DATA" = product ]
then
  echo "product"
  

  if [ "$rot_only" = 1 ]
  then
    Epoch=100
    save_step=10
    Model_LIST=`seq  10 10 100`

  fi 
fi 



target=results_random_crop_cluster/Deep_Metric
if [ "$rot" = 0 ]
then
  echo "change target"
  target=results_random_crop_cluster/reproduce
elif [ "$rot_only" = 1 ]
then
    echo "change target"
    target=rot_only
fi
if [ "$NET" != Inception ]
then
    echo "change target"
    target=results_random_crop_cluster/backbone
fi

if [ "$DIM" = 1000 ] || [ "$DIM" = 2000 ] || [ "$DIM" = 4000 ] || [ "$DIM" = 7000 ] || [ "$DIM" = 9000 ] || [ "$DIM" = 10000 ]
then
    echo "change target"
    target=cluster_only
    Epoch=100
    save_step=10
    Model_LIST=`seq  10 10 100`
fi 
if [ "$pretrained" = False ]
then
    echo "change target"
    target=scratch
    NET=ResNet18
    if [ "$DATA" = product ]
    then 
        Epoch=300
        save_step=25
        Model_LIST=`seq  25 25 300`
    fi 
    if [ "$DATA" = cifar ]
    then 
        Epoch=600
        save_step=50
        Model_LIST=`seq  50 50 600`
    fi
fi


root=/checkpoint/xuefeicao01/metric_learning/${target}
CHECKPOINTS=${root}/ckpt











# echo $rot
# echo $DATA
# echo $LOSS
# echo $ROT_LR
# echo $ALPHA
# echo $rot_bt
# echo $rot_batch
# echo $LR
# echo $DIM
# echo ${up_step}
# echo ${num_clusters}






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
#SAVE_DIR=${CHECKPOINTS}/${LOSS}/${DATA}/${NET}-DIM-${DIM}-lr-${LR}-ratio-${RATIO}-BatchSize-${BatchSize}-rot_lr-${ROT_LR}-self_supervision_rot-${rot}-rot_bt-${rot_bt}-rot_batch-${rot_batch}-ALPHA-${ALPHA}-BETA-${BETA}
#SAVE_DIR=${CHECKPOINTS}/${LOSS}/${DATA}/${NET}-DIM-${DIM}-lr-${LR}-ratio-${RATIO}-BatchSize-${BatchSize}-rot_lr-${ROT_LR}-self_supervision_rot-${rot}-rot_bt-${rot_bt}-ALPHA-${ALPHA}-BETA-${BETA}
if_exist_mkdir ${SAVE_DIR}
width=227




num_instances=5
if [ "$DATA" = cifar ]
    then 
      num_instances=50
    fi




# if [ ! -n "$1" ] ;then
# echo $width
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


# if [ "$rot" != 0 ] || [ "$target" == "cluster_only" ]
# then
CUDA_VISIBLE_DEVICES=0 python ../test_best_nmi.py --net ${NET} \
--data $DATA \
--net $NET \
--dim $DIM \
--save_txt ${save_txt} \
--data_root ${DATA_ROOT} \
--batch_size 100 \
-g_eq_q ${Gallery_eq_Query} \
--width ${width} \
--save_dir ${SAVE_DIR}\
| tee -a ${save_txt}
  
# fi



