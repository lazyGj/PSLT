# ------------pslt --------
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 21654 --use_env main.py --model PSLT --batch-size 256 --data-set IMNET --output_dir eval_IMNET1000/pslt/ --distillation-type none --aa rand-m9-mstd0.5-inc1 --spatial-attn dilated_local --teacher-model regnety_160 --data-path /data/imagenet --eval --resume ckpt/pslt.pth && echo fuck


# ------------plst-large ---
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 21654 --use_env main.py --model PSLT_large --batch-size 256 --data-set IMNET --output_dir eval_IMNET1000/pslt_large/ --distillation-type none --aa rand-m9-mstd0.5-inc1 --spatial-attn dilated_local --teacher-model regnety_160 --data-path /data/imagenet --eval --resume ckpt/pslt_large.pth && echo fuck


# ------------pslt-tiny ----
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 21654 --use_env main.py --model PSLT_tiny --batch-size 256 --data-set IMNET --output_dir eval_IMNET1000/pslt_tiny/ --distillation-type none --aa rand-m9-mstd0.5-inc1 --spatial-attn dilated_local --teacher-model regnety_160 --data-path /data/imagenet  --eval --resume ckpt/pslt_tiny.pth && echo fuck
