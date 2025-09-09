复现stanford_alpaca
搭建环境：
创建环境：conda create --name llama python=3.10
安装torch+cuda:pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
拉代码：git clone https://github.com/tatsu-lab/stanford_alpaca.git
下载llama-2-7b原版模型：https://huggingface.co/meta-llama/Llama-2-7b
将原版模型转化为hf版本：python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir path_to_original_llama_root_dir \
    --model_size 7B \
    --output_dir path_to_original_llama_hf_dir
安装依赖包：pip install -r requirements.txt
可能会报错：AttributeError: 'FullyShardedDataParallelPlugin' object has no attribute 'activation_checkpointing' 修改依赖：https://github.com/huggingface/transformers/issues/25988
  解决方法：pip install git+https://github.com/huggingface/transformers
  也可用简单方法，参考下面，将dataclasses.py替换一下
      The 'FullyShardedDataParallelPlugin' class in accelerate version v0.22.0 does not have 'activation_checkpointing'. but the main branch does.
      v0.22.0
        https://github.com/huggingface/accelerate/blob/6b3e559926afc4b9a127eb7762fc523ea0ea656a/src/accelerate/utils/dataclasses.py#L778
      main
        https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/utils/dataclasses.py#L783
关闭wandb相关：
    不配做wandb会报错，可直接关闭，将/root/anaconda3/envs/pytorch/lib/python3.10/site-packages/transformers/integrations/integration_utils.py中WandbCallback中的函数on_train_begin、on_train_end等直接return就好
开始训练：
    torchrun --nproc_per_node=4 --master_port=1234 train.py \
    --model_name_or_path /home/workspace/d00029379/project/stanford_alpaca-main/llama_2_7b_hf \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir stanford_alpaca_2_7b \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True


## 安装openrlhf踩的坑

1.遇到依赖flash_attn安装失败的问题：https://github.com/Dao-AILab/flash-attention/issues/1799
参考peterroelants的回答：
Using pip install "flash_attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" worked for me (after installing pytorch 2.8 with CUDA 12.8 support)

### 解决方法：

1.查看环境torch和cuda版本号：2.8.0+cu128

2.查到https://github.com/Dao-AILab/flash-attention/releases中的对应版本，这里下载的https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

3.本地安装：pip install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
