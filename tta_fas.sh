# 导入动态链接库 (解决可能会报的 libstdc++.so 找不到等问题)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

NPROC_PER_NODE=2
DEVICES="0,1"
CLIP_PATH="/share/rongss/test_time/IAPL/models/clip/ViT-L-14.pt"

# 1. 禁用 P2P (Peer-to-Peer) 通信。V100/容器环境下最容易引发段错误的原因。
export NCCL_P2P_DISABLE=1

# 2. 禁用 InfiniBand 网络，强制走普通网卡。
export NCCL_IB_DISABLE=1

# 3. 开启 NCCL 报错日志（可选，如果你想看到更详细的底层网络日志）
# export NCCL_DEBUG=INFO

if [ "$NPROC_PER_NODE" -le 1 ]; then
    CUDA_VISIBLE_DEVICES=0 WORLD_SIZE= RANK= LOCAL_RANK= python main.py \
        --batchsize 32 \
        --evalbatchsize 32 \
        --clip_path "${CLIP_PATH}" \
        --dataset_path "/share/rongss/test_time/IAPL/Datasets/FasImage" \
        --train_selected_subsets 'OULU_NPU' \
        --test_selected_subsets 'MSU_MFSD' 'CASIA_FASD' 'ReplayAttack'\
        --lr 0.005 \
        --model_name tta\
        --dataset FasImage \
        --epoch 1 \
        --lr_drop 10 \
        --gate True \
        --condition True \
        --pretrained_model /share/rongss/test_time/IAPL/results/fas_oulo/checkpoint_best_hter.pth \
        --eval \
        --smooth True\
        --tta True \
        --tta_steps 2 \
        --ois True \
        --num_workers 8
else
    CUDA_VISIBLE_DEVICES=${DEVICES} torchrun \
        --nproc_per_node=${NPROC_PER_NODE} \
        --master_port 29578 \
        main.py \
    --batchsize 32 \
    --evalbatchsize 32 \
    --clip_path "${CLIP_PATH}" \
    --dataset_path "/share/rongss/test_time/IAPL/Datasets/FasImage" \
    --train_selected_subsets 'OULU_NPU' \
    --test_selected_subsets 'MSU_MFSD' 'CASIA_FASD' 'ReplayAttack'\
    --lr 0.005 \
    --model_name tta\
    --dataset FasImage \
    --epoch 1 \
    --lr_drop 10 \
    --gate True \
    --condition True \
    --pretrained_model /share/rongss/test_time/IAPL/results/fas_oulo/checkpoint_best_hter.pth \
    --eval \
    --smooth True\
    --tta True \
    --tta_steps 2 \
    --ois True \
    --num_workers 8
fi