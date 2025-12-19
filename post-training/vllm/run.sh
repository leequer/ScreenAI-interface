model_path="Qwen/Qwen2.5-VL-3B-Instruct"
save_name=test.json

cd ../LL*/
export DISABLE_VERSION_CHECK=1
python scripts/vllm_infer.py --model_name_or_path $model_path --template qwen2_vl --dataset websight_test --cutoff_len 8192 --max_new_tokens 8192 --temperature 0 --image_max_pixels 1843200 --save_name $save_name