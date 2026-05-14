# Load shared libs from the active conda env if present.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

NPROC_PER_NODE=1
DEVICES="0"
CLIP_PATH="/share/rongss/test_time/IAPL/models/clip/ViT-L-14.pt"

# Disable P2P and InfiniBand to avoid NCCL issues in some container/V100 setups.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

if [ "$NPROC_PER_NODE" -le 1 ]; then
    CUDA_VISIBLE_DEVICES=${DEVICES} WORLD_SIZE= RANK= LOCAL_RANK= python main.py \
        --batchsize 32 \
        --evalbatchsize 32 \
        --clip_path "${CLIP_PATH}" \
        --dataset_path "/share/rongss/test_time/IAPL/Datasets/FasImage" \
        --train_selected_subsets 'OULU_NPU' \
        --test_selected_subsets 'MSU_MFSD' 'CASIA_FASD' 'ReplayAttack' \
        --lr 0.00005 \
        --model_name fas_oulo \
        --dataset FasImage \
        --epoch 5 \
        --lr_drop 10 \
        --gate True \
        --condition True \
        --smooth True
else
    CUDA_VISIBLE_DEVICES=${DEVICES} torchrun \
        --nproc_per_node=${NPROC_PER_NODE} \
        --master_port 29581 \
        main.py \
        --batchsize 32 \
        --evalbatchsize 32 \
        --clip_path "${CLIP_PATH}" \
        --dataset_path "/share/rongss/test_time/IAPL/Datasets/FasImage" \
        --train_selected_subsets 'OULU_NPU' \
        --test_selected_subsets 'MSU_MFSD' 'CASIA_FASD' 'ReplayAttack' \
        --lr 0.00005 \
        --model_name fas_oulo \
        --dataset FasImage \
        --epoch 5 \
        --lr_drop 10 \
        --gate True \
        --condition True \
        --smooth True
fi
