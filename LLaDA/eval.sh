# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
model_path=output_model/dspark_llada

############################################### gsm8k ###############################################
accelerate launch eval_llada.py --model dspark_llada --include_path tasks --tasks dspark_gsm8k --num_fewshot 0 --batch_size 1 \
--model_args model_path=${model_path},gen_length=256,block_length=32,boxed=True \
--output_path evals_results/gsm8k --log_samples

############################################### minerva_math ###############################################
accelerate launch eval_llada.py --model dspark_llada --include_path tasks --tasks dspark_minerva_math --num_fewshot 4 --batch_size 1 \
--model_args model_path=${model_path},gen_length=256,block_length=32,boxed=True,block_add_threshold=0.1,decoded_token_threshold=0.95 \
--output_path evals_results/minerva_math --log_samples

############################################### humaneval ###############################################
accelerate launch eval_llada.py --model dspark_llada --include_path tasks --tasks dspark_humaneval --num_fewshot 0 --batch_size 1 \
--confirm_run_unsafe_code --model_args model_path=${model_path},gen_length=256,block_length=32,factor=0.5 \
--output_path evals_results/humaneval --log_samples

############################################### mbpp ###############################################
accelerate launch eval_llada.py --model dspark_llada --include_path tasks --tasks dspark_mbpp --num_fewshot 3 --batch_size 1 \
--confirm_run_unsafe_code --model_args model_path=${model_path},gen_length=256,block_length=32 \
--output_path evals_results/mbpp --log_samples
