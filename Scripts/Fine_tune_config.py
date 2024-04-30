
## Modified from the original lightning_sam script to work with the current directory structure

import os
import torch
from box import Box

# Check if the code is running on Google Colab
on_colab = 'COLAB_GPU' in os.environ


current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

image_dir =  os.path.join(parent_dir, 'Demo\Train_data')
out_dir = os.path.join(parent_dir, 'Checkpoints_models\out')
checkpoint = os.path.join(parent_dir, 'Checkpoints_models\sam_vit_h_4b8939.pth')
annotation_val = os.path.join(parent_dir, 'Demo\Train_data\annotations_val.json')
annotation_train =  os.path.join(parent_dir, 'Demo\Train_data\annotations_train.json')

    
config = {
    "num_devices": torch.cuda.device_count(),
    "batch_size": 1,
    "num_workers": 2,
    "num_epochs": 1,
    "eval_interval": 2,
    "out_dir": out_dir,
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_h',
        "checkpoint": checkpoint,
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": image_dir,
            "annotation_file": annotation_train
        },
        "val": {
            "root_dir": image_dir,
            "annotation_file": annotation_val
        }
    }
}

ft_cfg = Box(config)