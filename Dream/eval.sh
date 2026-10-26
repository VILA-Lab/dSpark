# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
model_path=output_model/dspark_dream

############################################### gsm8k ###############################################
accelerate launch eval_dream.py --model dspark_dream --include_path tasks --tasks gsm8k_cot_zeroshot --num_fewshot 0 --batch_size 1 \
--apply_chat_template --model_args model_path=${model_path},gen_length=256,block_length=32 \
--output_path evals_results/gsm8k --log_samples

############################################### minerva_math ###############################################
accelerate launch eval_dream.py --model dspark_dream --include_path tasks --tasks dspark_minerva_math --num_fewshot 0 --batch_size 1 \
--apply_chat_template --model_args model_path=${model_path},gen_length=256,block_length=32 \
--output_path evals_results/minerva_math --log_samples

############################################### humaneval ###############################################
accelerate launch eval_dream.py --model dspark_dream --include_path tasks --tasks dspark_humaneval_instruct --num_fewshot 0 --batch_size 1 \
--apply_chat_template --confirm_run_unsafe_code --model_args model_path=${model_path},gen_length=256,block_length=32 \
--output_path evals_results/humaneval --log_samples

############################################### mbpp ###############################################
accelerate launch eval_dream.py --model dspark_dream --include_path tasks --tasks dspark_mbpp_instruct --num_fewshot 0 --batch_size 1 \
--apply_chat_template --confirm_run_unsafe_code --model_args model_path=${model_path},gen_length=256,block_length=32 \
--output_path evals_results/mbpp --log_samples
