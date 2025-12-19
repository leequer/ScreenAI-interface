[TOC]



# LLama Factory

conda create -n sft python=3.10

pip install -e ".[metrics,torch]"

pip install deepspeed

cd data

bash get_data.sh

cd ..

bash run_exp.sh



# VLM-R1

conda create -n rl python=3.10

bash setup.sh

pip install json_repair

bash run_exp.sh

# vLLM
conda create -n vllm python=3.10

pip install vllm

