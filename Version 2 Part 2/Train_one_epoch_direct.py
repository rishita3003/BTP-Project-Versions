# -*- coding: utf-8 -*-
"""
Direct training mode without reconstruction objectives.
Only uses ControlGAN attention + segmentation + contrastive loss.

Loss: L_total = L_seg + δ * L_contrast

Exports: train_one_epoch_direct, validate_direct, print_summary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_contrastive_loss(visual_feat, text_feat, temperature=0.07):
    """
    Global contrastive loss for text-image alignment.
    """
    # Pool spatial dimensions if needed
    if len(visual_feat.shape) == 4:  # (B, C, H, W)
        visual_feat = visual_feat.mean(dim=[2, 3])  # (B, C)
    elif len(visual_feat.shape) == 3:  # (B, N, C)
        visual_feat = visual_feat.mean(dim=1)  # (B, C)
    
    if len(text_feat.shape) == 3:  # (B, N, C)
        text_feat = text_feat.mean(dim=1)  # (B, C)
    
    # Normalize features
    visual_feat = F.normalize(visual_feat, dim=-1, p=2)
    text_feat = F.normalize(text_feat, dim=-1, p=2)
    
    # Compute similarity matrix
    batch_size = visual_feat.shape[0]
    logits = torch.matmul(visual_feat, text_feat.T) / temperature  # (B, B)
    
    # Labels (diagonal elements are positive pairs)
    labels = torch.arange(batch_size, device=visual_feat.device)
    
    # Bidirectional contrastive loss
    loss_v2t = F.cross_entropy(logits, labels)
    loss_t2v = F.cross_entropy(logits.T, labels)
    
    return (loss_v2t + loss_t2v) / 2.0


def unpack_batch_data(batch_data, batch_idx, logger):
    """
    Unified batch data unpacking that handles all possible formats.
    Returns: imgs, masks, text_token, text_mask
    """
    try:
        # Case 1: Tuple/list containing dict as first element
        if isinstance(batch_data, (list, tuple)) and len(batch_data) > 0 and isinstance(batch_data[0], dict):
            batch_dict = batch_data[0]
            imgs = batch_dict['image']
            masks = batch_dict['label']
            text_token = batch_dict.get('text_token', None)
            text_mask = batch_dict.get('text_mask', None)
            
            if batch_idx == 0:
                logger.info(f"✓ Data format: TUPLE containing DICT with keys {list(batch_dict.keys())}")
            return imgs, masks, text_token, text_mask
        
        # Case 2: Direct dictionary
        elif isinstance(batch_data, dict):
            imgs = batch_data['image']
            masks = batch_data['label']
            text_token = batch_data.get('text_token', None)
            text_mask = batch_data.get('text_mask', None)
            
            if batch_idx == 0:
                logger.info(f"✓ Data format: DICT with keys {list(batch_data.keys())}")
            return imgs, masks, text_token, text_mask
        
        # Case 3: Tuple/list of tensors
        elif isinstance(batch_data, (list, tuple)):
            imgs = batch_data[0]
            masks = batch_data[1]
            text_token = batch_data[2] if len(batch_data) > 2 else None
            text_mask = batch_data[3] if len(batch_data) > 3 else None
            
            if batch_idx == 0:
                logger.info(f"✓ Data format: TUPLE of {len(batch_data)} tensors")
            return imgs, masks, text_token, text_mask
        
        else:
            if batch_idx == 0:
                logger.info(f"ERROR: Unexpected batch_data type: {type(batch_data)}")
            return None, None, None, None
    
    except Exception as e:
        if batch_idx == 0:
            logger.info(f"ERROR unpacking batch: {e}")
        return None, None, None, None


def train_one_epoch_direct(config, train_loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger):
    """
    Training loop for direct segmentation mode (without reconstruction).
    
    Loss: L_total = α * L_seg + δ * L_contrast
    """
    model.train()
    
    # Handle DataParallel - disable reconstruction
    if isinstance(model, nn.DataParallel):
        model.module.aux = False
    else:
        model.aux = False
    
    epoch_loss = 0
    epoch_dice = 0
    num_batches = 0
    
    for batch_idx, batch_data in enumerate(train_loader):
        
        # Unpack data
        imgs, masks, text_token, text_mask = unpack_batch_data(batch_data, batch_idx, logger)
        
        if imgs is None or masks is None:
            continue
        
        # Verify we have valid tensors
        if not isinstance(imgs, torch.Tensor) or not isinstance(masks, torch.Tensor):
            if batch_idx == 0:
                logger.info(f"ERROR: imgs or masks not tensors")
            continue
        
        # Move to GPU
        imgs = imgs.cuda()
        masks = masks.cuda()
        
        if text_token is not None and isinstance(text_token, torch.Tensor):
            text_token = text_token.cuda()
        if text_mask is not None and isinstance(text_mask, torch.Tensor):
            text_mask = text_mask.cuda()
        
        # Forward pass
        try:
            if text_token is not None and text_mask is not None:
                logits, loss_dic = model(imgs, masks, text_token, text_mask, mode="train")
            else:
                if batch_idx == 0:
                    logger.info("WARNING: No text data, using images only")
                logits = model(imgs)
                loss_dic = {}
        except Exception as e:
            logger.info(f"ERROR in forward pass batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Compute segmentation loss
        if config.n_labels == 1:
            loss_seg = criterion(logits, masks.float())
        else:
            loss_seg = criterion(logits, masks.squeeze(1).long())
        
        # Ensure loss is scalar (mean reduction if not already)
        if loss_seg.dim() > 0:
            loss_seg = loss_seg.mean()
        
        # Total loss (use .get() for safe access)
        loss_weight = getattr(config, 'loss_weight', {})
        seg_weight = loss_weight.get('loss_criterion', 1.0) if isinstance(loss_weight, dict) else 1.0
        loss_total = seg_weight * loss_seg
        
        # Add contrastive loss if enabled
        # NOTE: Contrastive loss is disabled in direct mode because it requires
        # access to intermediate encoder features (512-dim) rather than final logits (1-dim).
        # The original RecLMIS uses reconstruction-based alignment instead.
        # To enable contrastive loss, modify RecLMIS.forward() to return encoder features.
        contrastive_loss = torch.tensor(0.0, device=imgs.device)
        
        # Contrastive loss disabled - dimension mismatch between logits (B,1,H,W) and text (B,N,512)
        # if getattr(config, 'use_contrastive', False) and text_token is not None:
        #     ... requires encoder features, not logits
        
        # Backward and optimize
        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Compute dice score
        with torch.no_grad():
            if config.n_labels == 1:
                pred_binary = (logits > 0.5).float()
            else:
                pred_binary = torch.argmax(logits, dim=1, keepdim=True).float()
            
            # Dice = 2 * |A ∩ B| / (|A| + |B|)
            # Compute per-sample then average
            pred_flat = pred_binary.view(pred_binary.size(0), -1)
            mask_flat = masks.float().view(masks.size(0), -1)
            
            intersection = (pred_flat * mask_flat).sum(dim=1)
            union = pred_flat.sum(dim=1) + mask_flat.sum(dim=1)
            
            # Per-sample dice, then mean across batch
            dice_per_sample = (2.0 * intersection + 1e-8) / (union + 1e-8)
            dice_score = dice_per_sample.mean().item()
        
        # Accumulate metrics
        epoch_loss += loss_total.item()
        epoch_dice += dice_score  # Already a float from .item() above
        num_batches += 1
        
        # Logging
        print_freq = getattr(config, 'print_frequency', 10)
        if batch_idx % print_freq == 0:
            logger.info(
                f'   [Train] Epoch: [{epoch+1}][{batch_idx+1}/{len(train_loader)}]  '
                f'Loss: {loss_total.item():.4f} '
                f'Seg: {loss_seg.item():.4f} '
                f'Contrast: {contrastive_loss.item():.4f} '
                f'Dice: {dice_score:.4f}'
            )
        
        # Tensorboard logging
        if writer is not None:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss_Total', loss_total.item(), step)
            writer.add_scalar('Train/Loss_Segmentation', loss_seg.item(), step)
            writer.add_scalar('Train/Loss_Contrastive', contrastive_loss.item(), step)
            writer.add_scalar('Train/Dice', dice_score, step)
    
    # Calculate averages
    if num_batches == 0:
        logger.info("ERROR: No batches processed!")
        return 0.0, 0.0
        
    average_loss = epoch_loss / num_batches
    average_dice = epoch_dice / num_batches
    
    logger.info(
        f'\n   Epoch [{epoch+1}/{config.epochs}] Training Summary (DIRECT MODE):\n'
        f'   Average Loss: {average_loss:.4f}\n'
        f'   Average Dice: {average_dice:.4f}\n'
    )
    
    return average_loss, average_dice


def validate_direct(config, val_loader, model, criterion, logger):
    """
    Validation loop for direct mode.
    Returns: (val_loss, val_dice)
    """
    model.eval()
    
    # Disable reconstruction for validation
    if isinstance(model, nn.DataParallel):
        model.module.aux = False
    else:
        model.aux = False
    
    val_loss = 0
    val_dice = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            
            # Unpack data
            imgs, masks, text_token, text_mask = unpack_batch_data(batch_data, batch_idx, logger)
            
            if imgs is None or masks is None:
                continue
            
            # Move to GPU
            imgs = imgs.cuda()
            masks = masks.cuda()
            
            if text_token is not None and isinstance(text_token, torch.Tensor):
                text_token = text_token.cuda()
            if text_mask is not None and isinstance(text_mask, torch.Tensor):
                text_mask = text_mask.cuda()
            
            # Forward pass
            try:
                if text_token is not None and text_mask is not None:
                    logits, _ = model(imgs, masks, text_token, text_mask, mode="train")
                else:
                    logits = model(imgs)
            except Exception as e:
                if batch_idx == 0:
                    logger.info(f"ERROR in validation forward pass: {e}")
                continue
            
            # Compute loss
            if config.n_labels == 1:
                loss = criterion(logits, masks.float())
            else:
                loss = criterion(logits, masks.squeeze(1).long())
            
            # Ensure loss is scalar
            if loss.dim() > 0:
                loss = loss.mean()
            
            # Compute dice
            if config.n_labels == 1:
                pred_binary = (logits > 0.5).float()
            else:
                pred_binary = torch.argmax(logits, dim=1, keepdim=True).float()
            
            # Dice = 2 * |A ∩ B| / (|A| + |B|)
            pred_flat = pred_binary.view(pred_binary.size(0), -1)
            mask_flat = masks.float().view(masks.size(0), -1)
            
            intersection = (pred_flat * mask_flat).sum(dim=1)
            union = pred_flat.sum(dim=1) + mask_flat.sum(dim=1)
            
            dice_per_sample = (2.0 * intersection + 1e-8) / (union + 1e-8)
            dice_score = dice_per_sample.mean().item()
            
            val_loss += loss.item()
            val_dice += dice_score
            num_batches += 1
    
    if num_batches == 0:
        logger.info("ERROR: No validation batches processed!")
        return 0.0, 0.0
    
    avg_loss = val_loss / num_batches
    avg_dice = val_dice / num_batches
    
    logger.info(f'   Validation - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}')
    
    return avg_loss, avg_dice


def print_summary(epoch, lr, train_loss, train_dice, val_loss, val_dice, time_elapsed, logger):
    """Print epoch summary."""
    logger.info(
        f'\n{"="*70}\n'
        f'Epoch {epoch} Summary:\n'
        f'{"="*70}\n'
        f'Learning Rate: {lr:.6f}\n'
        f'Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}\n'
        f'Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}\n'
        f'Time Elapsed: {time_elapsed:.2f}s\n'
        f'{"="*70}'
    )
