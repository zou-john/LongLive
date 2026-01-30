torchrun \
  --nproc_per_node=1 \
  --master_port=29500 \
  streaming_inference.py \
  --config_path configs/longlive_streaming_inference.yaml