import os
import subprocess
import yaml
from datetime import datetime

# 运行阶段
DO_SFT_TRAIN = True
DO_EVAL = False
# 模型配置
MODEL_PATH = "/root/autodl-tmp"
MODEL_NAME = "llama3-8b"
# 数据集配置
SFT_DATASET = [{
    "name": "alpaca_en_demo",
    "lang": "en",
}]
EVAL_DATASETS = [
    {
        "name": "mmlu_test",
        "lang": "en",
    },
    {
        "name": "ceval_validation",
        "lang": "zh",
    },
    {
        "name": "cmmlu_test",
        "lang": "zh",
    }
]
EVAL_BATCHSIZE = 4
# 超参
# FINETUNE_TYPE = "origin"
FINETUNE_TYPE = "lora"
# lora_rank = 8
# learning_rate = 1e-4
# config_str = ""
cfg = {
    "config_str": "",
    "learning_rate": 1e-4,
    "lora_rank": 8,
}

# def print_cfg():
#     for (key, value) in cfg:
#         print(f"{key} = {value}")

def set_train_config_file(dataset_config):
    # 创建时间戳用于唯一标识
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 配置参数
    config ={}
    
    # model
    config.update({
        "model_name_or_path": f"{MODEL_PATH}/meta-llama/Meta-Llama-3-8B-Instruct",
        "trust_remote_code": False,
    })    
    
    # finetune method
    if FINETUNE_TYPE == "lora":
        config.update({
            "stage": "sft",
            "do_train": True,
            "finetuning_type": FINETUNE_TYPE,
            "lora_rank": cfg['lora_rank'],
            "lora_target": "all",
        })
        
    if FINETUNE_TYPE == "full":
        config.update({
            "stage": "sft",
            "do_train": True,
            "finetuning_type": FINETUNE_TYPE,
            ":deepspeed": ":examples/deepspeed/ds_z3_config.json"  # choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]":
        })       
        
    # dataset
    config.update({
        "dataset": f"identity,{dataset_config['name']}",
        "template": "llama3",
        "cutoff_len": 2048,
        "max_samples": 1000,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        "dataloader_num_workers": 4,
    })
    
    # output
    config.update({
        "output_dir": f"{MODEL_PATH}/saves/{MODEL_NAME}/{cfg['config_str']}",
        "logging_steps": 10,
        "save_steps": 500,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "save_only_model": False,
        "report_to": "none"  # choices: [none, wandb, tensorboard, swanlab, mlflow]
    })
    
    ### train
    config.update({
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": cfg['learning_rate'],
        "num_train_epochs": 3.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "ddp_timeout": 180000000,
        "resume_from_checkpoint": None,
    })
    
    # 创建配置文件
    config_file = f"configs/train/{MODEL_NAME}/{cfg['config_str']}/{dataset_config['name']}_{timestamp}.yaml"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    with open(config_file, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    
    return config_file


def set_merge_config_file(dataset_config):# TODO: merge
    """运行单个数据集的评估"""
    # 创建时间戳用于唯一标识
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 配置参数
    config ={}
    
    # model
    config.update({
        "model_name_or_path": f"{MODEL_PATH}/meta-llama/Meta-Llama-3-8B-Instruct",
        "trust_remote_code": False,
    })    
    
    # ### model
    # model_name_or_path: /root/autodl-tmp/meta-llama/Meta-Llama-3-8B-Instruct
    # adapter_name_or_path: /root/autodl-tmp/saves/llama3-8b/lora/sft
    # template: llama3
    # trust_remote_code: false

    # ### export
    # export_dir: /root/autodl-tmp/output/llama3_lora_sft
    # export_size: 5
    # export_device: cpu  # choices: [cpu, auto]
    # export_legacy_format: false
    
    # 创建配置文件
    config_file = f"configs/train/{MODEL_NAME}/{cfg['config_str']}/{dataset_config['name']}_{timestamp}.yaml"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    with open(config_file, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    
    return config_file


def set_eval_config_file(dataset_config):
    """运行单个数据集的评估"""
    # 创建时间戳用于唯一标识
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 合并配置
    config ={}
    
    # model
    config.update({
        "model_name_or_path": f"{MODEL_PATH}/meta-llama/Meta-Llama-3-8B-Instruct",
        "trust_remote_code": True,
    })    
     
    if FINETUNE_TYPE == "lora":
        config.update({
            "adapter_name_or_path": f"{MODEL_PATH}/saves/{MODEL_NAME}/{cfg['config_str']}"
        })        
        
    # dataset
    config.update({
        "template": "fewshot",
        "n_shot": 5,
    })
    config.update({
        "task": dataset_config['name'],
        "lang": dataset_config['lang'],
    })
    
    # output
    config.update({
        "save_dir": f"saves/{MODEL_NAME}/{cfg['config_str']}/{dataset_config['name']}",
    })
    print("save_dir:", f"saves/{MODEL_NAME}/{cfg['config_str']}/{dataset_config['name']}")
    
    # eval
    config.update({
        "batch_size": EVAL_BATCHSIZE
    })
    
    # 创建配置文件
    config_file = f"configs/eval/{MODEL_NAME}/{cfg['config_str']}/{dataset_config['name']}_{timestamp}.yaml"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    with open(config_file, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    
    return config_file

def run_llf(mode: str, dataset):

    if mode == "train":
        print(f"⏳ 开始后训练 {dataset['name']}...")
        config_file = set_train_config_file(dataset)

    if mode == "eval":
        print(f"⏳ 开始评估 {dataset['name']}...")
        config_file = set_eval_config_file(dataset)
    
    cmd = [
        "llamafactory-cli", mode, 
        config_file
    ]
    print(f"📄 配置文件: {config_file}")

    # 运行评估        
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 检查执行结果
    if result.returncode == 0:
        print(f"✅ {dataset['name']} {mode} finished!")
        # print(f"📁 结果保存到: {config['output']['save_dir']}")
    else:
        print(f"❌ {dataset['name']} {mode} failed!")
        print("错误信息:")
        print(result.stderr)
        
    return result.returncode == 0
        

def train_and_eval():
    cfg['config_str'] = f"{FINETUNE_TYPE}_lr{cfg['learning_rate']}"
    if FINETUNE_TYPE == "lora":
        cfg['config_str'] = f"{cfg['config_str']}_rank{cfg['lora_rank']}"
    print(cfg)


    if DO_SFT_TRAIN:# sft TRAIN
        mode = "train"
        print("=" * 50)
        print(f"🚀 开始自动化微调 {MODEL_NAME}")
        print("=" * 50)    
        
        dataset = SFT_DATASET[0]
        # print(dataset)
        success = run_llf(mode, dataset)
        if not success:
            print(f"⚠️ 训练失败，实验参数{cfg['config_str']}")
            return
    
    if DO_EVAL:# EVAL
        print("=" * 50)
        print(f"🚀 开始自动化评估 {MODEL_NAME}")
        print(f"📊 数据集数量: {len(EVAL_DATASETS)}")
        print("=" * 50)  
        
        mode = "eval"
        # 顺序评估所有数据集
        for dataset in EVAL_DATASETS:
            success = run_llf(mode, dataset)
            if not success:
                print(f"⚠️ 终止评估流程，因为 {dataset['name']} 失败，实验参数{cfg['config_str']}")
                break
        
        print("=" * 50)
        print("🏁 所有评估任务完成!")


os.makedirs("configs", exist_ok=True)

cfg['learning_rate'] = 1e-5
for _ in range(3):
    cfg['lora_rank'] = 8
    for _ in range(3):
        train_and_eval()
        
        cfg['lora_rank'] *= 2
    cfg['learning_rate'] *= 10
    
# def main():
    # 创建配置目录

# if __name__ == "__main__":
#     main()