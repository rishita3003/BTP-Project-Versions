# -*- coding: utf-8 -*-
import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
use_cuda = torch.cuda.is_available()
seed = 3407
os.environ['PYTHONHASHSEED'] = str(seed)

lr = 'cosineLR'  # Use cosineLR, exp, cosine, poly
n_channels = 3
n_labels = 1  
epochs = 2000
img_size = 224
print_frequency = 1
save_frequency = 5000
vis_frequency = 5000
early_stopping_patience = 150

pretrain = False
task_name = 'MoNuSeg' 
token_len = 10
learning_rate = 1e-3 

batch_size = 24

optimizer = "Adam"
weight_decay = 1e-5

model_name = 'RecLMIS'

# ====================================================================
# 🆕 CONTROLGAN INTEGRATION - NEW PARAMETERS
# ====================================================================
use_controlgan_attention = True  # Set to False to use original RecLMIS
training_mode = 'reconstruction'  # Options: 'reconstruction' or 'direct'

# ControlGAN Attention Settings
controlgan_num_heads = 8
controlgan_channel_reduction = 16

# Training mode specific settings
if training_mode == 'direct':
    # Direct segmentation mode (no reconstruction)
    use_contrastive = True
    contrastive_weight = 0.5  # delta in paper
    loss_weight = {
        "loss_criterion": 1.0,  # Only segmentation loss is primary
        "loss_ccl": 0.0,        # Disable CCL in direct mode
        "loss_text_rec": 0.0,   # Disable reconstruction
        "loss_img_rec": 0.0,
    }
else:
    # Reconstruction mode (original RecLMIS with ControlGAN attention)
    use_contrastive = False
    loss_weight = { 
        "loss_criterion": 5,    # alpha: segmentation weight
        "loss_ccl": 0.2,        # delta: contrastive weight
        "loss_text_rec": 1,     # beta: text reconstruction weight
        "loss_img_rec": 1,      # gamma: image reconstruction weight
    }
# ====================================================================

text_name = "text_alpha.xlsx"
resume = False

train_dataset = '/content/drive/MyDrive/RecLMIS/datasets/' + task_name + '/Train_Folder/'
val_dataset = '/content/drive/MyDrive/RecLMIS/datasets/' + task_name + '/Val_Folder/'
test_dataset = '/content/drive/MyDrive/RecLMIS/datasets/' + task_name + '/Test_Folder/'

# Update session name to include training mode
session_name = 'session' + '_' + training_mode + '_' + ('CG' if use_controlgan_attention else 'orig') + '_' + time.strftime('%m.%d_%Hh%M')
save_path = task_name + '/' + model_name + '/' + session_name + '/'
model_path = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path = save_path + session_name + ".log"
visualize_path = save_path + 'visualize_val/'

##########################################################################
# ViT configs
##########################################################################
def get_ViT_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.base_channel = 64 
    config.clip_backbone = "ViT-B/32"
    config.text_mask_rate = 0.3
    config.img_mask_rate = 0.5
    config.pool_mode = "max_pool"  # max_pool, aver_pool
    config.rec_trans_num_layers1 = 3
    config.mask_mode = "dist"
    config.frozen_clip = True
    config.mask_mode_dist_random = True
    config.dropout = True
    config.dropout_value = 0.5
    
    # ====================================================================
    # 🆕 CONTROLGAN SETTINGS - ADD TO ViT CONFIG
    # ====================================================================
    config.use_controlgan_attention = use_controlgan_attention
    config.training_mode = training_mode
    config.controlgan_num_heads = controlgan_num_heads
    config.controlgan_channel_reduction = controlgan_channel_reduction
    config.use_contrastive = use_contrastive if training_mode == 'direct' else False
    config.contrastive_weight = contrastive_weight if training_mode == 'direct' else 0.0
    # ====================================================================
    
    return config

# used in testing phase, copy the session name in training phase
test_session = "session_08.20_20h40" 
test_vis = False