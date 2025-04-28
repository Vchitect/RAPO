srun -p video-aigc-4 -n1 --gres=gpu:1 --cpus-per-task=8 --quotatype=spot --async \
-N1 --job-name=python \
python rewrite_via_instruction.py \
--input_path "./data/test_prompts.txt" \
--output_path "./output/rewrite_via_instruction/test_prompts.txt" \