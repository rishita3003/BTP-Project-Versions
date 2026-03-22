# -*- coding: utf-8 -*-
"""
RecLMIS Training Script with ControlGAN Attention
Supports both reconstruction and direct training modes.

Usage:
    python train_model.py -c Config_covid19 -g 0 -tm direct
    python train_model.py -c Config_covid19 -g 0 -tm reconstruction
"""

import argparse
import os
import math

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--cfg_path', '-c', default='Config_covid19', metavar='CFG_PATH',
                    type=str, help='Path to the config file')
parser.add_argument('--gpu', '-g', default='0', metavar='cuda',
                    type=str, help='device id')
parser.add_argument('--training_mode', '-tm', default=None, metavar='TRAINING_MODE',
                    type=str, choices=['reconstruction', 'direct'],
                    help='Training mode: reconstruction or direct (overrides config if specified)')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.cfg_path == "Config_monuseg":
    import Config_monuseg as config
elif args.cfg_path == "Config_Kvasir_Clinic":
    import Config_MosMedPlus as config
else:
    import Config_covid19 as config

# Override config if command line arg provided
if args.training_mode is not None:
    config.training_mode = args.training_mode
    print(f"Training mode overridden to: {args.training_mode}")

# Ensure training_mode exists in config
if not hasattr(config, 'training_mode'):
    config.training_mode = 'reconstruction'
    print("Warning: training_mode not in config, defaulting to 'reconstruction'")

import torch.optim
import torch.nn as nn
import time
from tensorboardX import SummaryWriter
import numpy as np
import random
from torch.backends import cudnn
from Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D, LV2D
from nets.RecLMIS import RecLMIS
from torch.utils.data import DataLoader
import logging
from Train_one_epoch import train_one_epoch, print_summary
from Train_one_epoch_direct import train_one_epoch_direct, validate_direct
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE, WeightedDiceCE, read_text, read_text_LV, save_on_batch


def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def save_checkpoint(state, save_path):
    """Save checkpoint with proper DataParallel handling."""
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']
    best_model = state['best_model']
    model = state['model']

    if best_model:
        filename = save_path + 'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + 'latest_model.pth.tar'
    
    logger.info('\t Saving to {}'.format(filename))
    torch.save(state, filename)


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):
    # Get training mode
    training_mode = getattr(config, 'training_mode', 'reconstruction')
    use_controlgan = getattr(config, 'use_controlgan_attention', False)
    
    # Log training configuration
    logger.info('='*70)
    logger.info('TRAINING CONFIGURATION')
    logger.info('='*70)
    logger.info(f'Task: {config.task_name}')
    logger.info(f'Model: {config.model_name}')
    logger.info(f'Training Mode: {training_mode}')
    logger.info(f'Use ControlGAN Attention: {use_controlgan}')
    logger.info(f'Batch Size: {batch_size}')
    logger.info(f'Learning Rate: {config.learning_rate}')
    logger.info(f'Epochs: {config.epochs}')
    logger.info('Loss Weights:')
    for key, val in config.loss_weight.items():
        logger.info(f'  {key}: {val}')
    logger.info('='*70)
    
    # Load train and val data
    train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    
    if config.task_name == 'MoNuSeg' or config.task_name == 'MosMedplus':
        print(f"Text file path : {config.train_dataset}Train_text.xlsx")
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')
        val_text = read_text(config.val_dataset + 'Val_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf,
                                       image_size=config.img_size, data_name=config.task_name, token_len=config.token_len, config=config, mode="train")
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size, data_name=config.task_name, token_len=config.token_len, config=config, mode="val")
    elif config.task_name == 'Covid19':
        text = read_text(config.task_dataset + 'Train_Val_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, text, train_tf,
                                       image_size=config.img_size, data_name=config.task_name, config=config, mode="train")
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, text, val_tf, image_size=config.img_size, data_name=config.task_name, config=config, mode="val")
    elif config.task_name == 'Kvasir_Clinic':
        text = read_text(config.train_dataset + '{}'.format(config.text_name))
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, text, train_tf,
                                       image_size=config.img_size, data_name=config.task_name, token_len=config.token_len, config=config, mode="train")
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, text, val_tf, image_size=config.img_size, data_name=config.task_name, token_len=config.token_len, config=config, mode="val")
    elif config.task_name == 'Kvasir_Clinic_Pra':
        text = read_text(config.train_dataset + '{}'.format(config.text_name))
        print('val_text: ',config.val_dataset + '{}'.format(config.text_name))
        val_text = read_text(config.val_dataset + '{}'.format(config.text_name))
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, text, train_tf,
                                       image_size=config.img_size, data_name=config.task_name, token_len=config.token_len, config=config, mode="train")
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size, data_name=config.task_name, token_len=config.token_len, config=config, mode="val")

    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              num_workers=8,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            worker_init_fn=worker_init_fn,
                            num_workers=8,
                            pin_memory=True)
                             
    lr = config.learning_rate
    logger.info(model_type)

    config_vit = config.get_ViT_config()
    model = RecLMIS(config, config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

    # Configure model based on training mode
    if training_mode == 'direct':
        model.aux = False
        logger.info('✓ Reconstruction disabled (direct training mode)')
    else:
        model.aux = True
        logger.info('✓ Reconstruction enabled (reconstruction training mode)')

    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    
    if config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        
    if config.lr == 'cosineLR':
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    elif config.lr == 'exp':
        lambda1 = lambda epoch: max(0.99**epoch, 0.1)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif config.lr == 'cosine':
        warm_up_steps = 0
        warm_up_with_cosine_lr = lambda step: step / warm_up_steps if step <= warm_up_steps and warm_up_steps!=0 else 0.5 * (math.cos((step - warm_up_steps) /(config.epochs - warm_up_steps) * math.pi) + 1)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    elif config.lr == 'poly':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(train_loader) * config.epochs)) ** 0.99)

    print(config.lr)
    
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: {}'.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    epoch = 0

    if config.resume:
        checkpoint = torch.load(config.resume_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=True)

    model = model.cuda()

    if config.resume:
        checkpoint = torch.load(config.resume_path, map_location='cpu')
        logger.info('resume path: {}'.format(config.resume_path))
        print(model.load_state_dict(checkpoint['state_dict']))
        
    if torch.cuda.device_count() > 1:
        logger.info("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    if config.resume:
        print(optimizer.load_state_dict(checkpoint['optimizer']))
        print(lr_scheduler.load_state_dict(checkpoint['lr_scheduler']))
        epoch = checkpoint['epoch']
        print("resume optimizer and lr scheduler successfuly")
    else:
        epoch = -999

    max_dice = 0.0
    best_epoch = 0
    
    for epoch in range(max(0, epoch+1), config.epochs):
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        
        # ================================================================
        # TRAINING
        # ================================================================
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        
        if training_mode == 'direct':
            logger.info('Using DIRECT training mode (ControlGAN attention only)')
            train_loss, train_dice = train_one_epoch_direct(
                config, train_loader, model, criterion, 
                optimizer, writer, epoch, None, model_type, logger
            )
        else:
            logger.info('Using RECONSTRUCTION training mode (ControlGAN + Reconstruction)')
            train_one_epoch(
                config, train_loader, model, criterion, 
                optimizer, writer, epoch, None, model_type, logger
            )
        
        # ================================================================
        # VALIDATION - Use correct function based on mode
        # ================================================================
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            
            if training_mode == 'direct':
                # Use validate_direct for direct mode
                val_loss, val_dice = validate_direct(
                    config, val_loader, model, criterion, logger
                )
            else:
                # Use train_one_epoch for reconstruction mode (with aux disabled)
                if isinstance(model, nn.DataParallel):
                    model.module.aux = False
                else:
                    model.aux = False
                    
                val_loss, val_dice = train_one_epoch(
                    config, val_loader, model, criterion,
                    optimizer, writer, epoch, lr_scheduler, model_type, logger
                )
                
                # Re-enable reconstruction after validation
                if isinstance(model, nn.DataParallel):
                    model.module.aux = True
                else:
                    model.aux = True
        
        # Update learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        # ================================================================
        # Save checkpoint (handle DataParallel properly)
        # ================================================================
        if isinstance(model, nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        
        # Save best model
        if val_dice > max_dice:
            if epoch + 1 > 0:
                logger.info(
                    '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice, val_dice))
                max_dice = val_dice
                best_epoch = epoch + 1
                save_checkpoint({
                    'epoch': epoch,
                    'best_model': True,
                    'model': model_type,
                    'state_dict': state_dict,
                    'val_loss': val_loss,
                    'optimizer': optimizer.state_dict(),
                    'training_mode': training_mode,
                    'use_controlgan': use_controlgan
                }, config.model_path)
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice, max_dice, best_epoch))
        
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))

        save_checkpoint({
            'epoch': epoch,
            'best_model': False,
            'model': model_type,
            'state_dict': state_dict,
            'val_loss': val_loss,
            "lr_scheduler": lr_scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
            'training_mode': training_mode,
            'use_controlgan': use_controlgan
        }, config.model_path)

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return model


if __name__ == '__main__':

    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)

    with open(args.cfg_path+'.py', 'r') as file:  
        lines = file.readlines()  
    for line in lines:  
        logger.info(line[:-1])

    model = main_loop(model_type=config.model_name, tensorboard=True)