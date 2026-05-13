NPROC_PER_NODE=${NPROC_PER_NODE:-4}
CLIP_PATH=${CLIP_PATH:-/path/to/ViT-L-14.pt}

if [ "$NPROC_PER_NODE" -le 1 ]; then
    WORLD_SIZE= RANK= LOCAL_RANK= python main.py \
        --batchsize 32 \
        --evalbatchsize 32 \
        --clip_path "${CLIP_PATH}" \
        --dataset_path "/share/rongss/test_time/IAPL/Datasets/GenImage" \
        --train_selected_subsets 'SDv14' \
        --test_selected_subsets 'BigGAN'\
        --lr 0.005 \
        --model_name tta\
        --dataset GenImage \
        --epoch 1 \
        --lr_drop 10 \
        --gate True \
        --condition True \
        --pretrained_model /share/rongss/test_time/IAPL/models/IAPL_pretrain/checkpoint_best_acc_sd14.pth \
        --eval \
        --smooth True\
        --tta True \
        --tta_steps 2 \
        --ois True \
        --num_workers 8
else
    python -m torch.distributed.launch \
        --nproc_per_node=${NPROC_PER_NODE} \
        --master_port 29578 \
        main.py \
    --batchsize 32 \
    --evalbatchsize 32 \
    --clip_path "${CLIP_PATH}" \
    --dataset_path "/share/rongss/test_time/IAPL/Datasets/GenImage" \
    --train_selected_subsets 'SDv14' \
    --test_selected_subsets 'BigGAN'\
    --lr 0.005 \
    --model_name tta\
    --dataset GenImage \
    --epoch 1 \
    --lr_drop 10 \
    --gate True \
    --condition True \
    --pretrained_model /share/rongss/test_time/IAPL/models/IAPL_pretrain/checkpoint_best_acc_sd14.pth \
    --eval \
    --smooth True\
    --tta True \
    --tta_steps 2 \
    --ois True \
    --num_workers 8
fi