
export PYTHONPATH=/mnt/afs_f/zhoudinghao/work/open/wenet/

year=2025
version=0.1
exp_name=ssl
base_dir=/mnt/afs_f/zhoudighao/maga/${year}/${version}/${exp_name}/

model_dir=${base_dir}/model
tensorboard_dir=${base_dir}/tensorboard

mkdir -p $model_dir
mkdir -p $tensorboard_dir

dist_backend=nccl
train_engine=torch_ddp # deepspeed or torch_fsdp
num_workers=10
prefetch=50

data_type=raw

train_config=/mnt/afs_f/zhoudinghao/maga/conf/stage1.yaml # bestrq/nestrq conf
train_data=...
cv_data=...

python3 ${PYTHONPATH}/wenet/bin/train.py \
        --train_engine ${train_engine} \
        --config ${train_config} \
        --data_type  ${data_type} \
        --train_data ${train_data} \
        --cv_data da ${cv_data} \
        ${checkpoint:+--checkpoint $checkpoint} \
        --model_dir ${model_dir} \
        --tensorboard_dir ${tensorboard_dir} \
        --ddp.dist_backend ${dist_backend} \
        --num_workers ${num_workers} \
        --prefetch ${prefetch} \
        --pin_memory \

