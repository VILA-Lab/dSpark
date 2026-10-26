# Build the dSpark training data with one process per GPU (run from this folder).
# Each process writes data/shard_<i>.jsonl; rerunning the script resumes unfinished shards.
MODEL_PATH=${MODEL_PATH:-Dream-org/Dream-v0-Instruct-7B}
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
mkdir -p data

for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} python data_prepare.py --model_path $MODEL_PATH \
    --num_shards ${#GPUS[@]} --shard_id $i --output_file data/shard_$i.jsonl > data/shard_$i.log 2>&1 &
done
wait
