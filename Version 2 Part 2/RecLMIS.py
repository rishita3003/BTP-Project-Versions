# -*- coding: utf-8 -*-
"""
RecLMIS with ControlGAN Attention Integration
Supports both reconstruction and direct training modes.
"""

from .controlgan_attention import EnhancedInteractor, SpatialAttentionModule, ChannelAttentionModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from .pixlevel import PixLevelModule
from .Interactor import Interactor
from .module_clip import CLIP, convert_weights, _PT_NAME
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, Slip
from .transformer import DualTransformer
from .transformer.mutihead_attention import MultiheadAttention
from .transformer.xpool import XPool


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pixModule = PixLevelModule(in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.pixModule(skip_x)
        x = torch.cat([skip_x_att, up], dim=1)
        return self.nConvs(x)


class RecLMIS(nn.Module):
    def __init__(self, global_config, config, n_channels=3, n_classes=1, img_size=224, vis=False):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.loss_weight = global_config.loss_weight
        
        # Encoder
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        
        # Interactor - choose based on config (check global_config first, then config)
        use_controlgan = getattr(global_config, 'use_controlgan_attention', False) or \
                         getattr(config, 'use_controlgan_attention', False)
        
        if use_controlgan:
            self.interact = EnhancedInteractor(
                config, vis, img_size=14, channel_num=512, patch_size=1, embed_dim=512
            )
            self.use_controlgan = True
            print("✓ Using EnhancedInteractor (ControlGAN attention)")
        else:
            self.interact = Interactor(
                config, vis, img_size=14, channel_num=512, patch_size=1, embed_dim=512
            )
            self.use_controlgan = False
            print("✓ Using original Interactor")
        
        # Decoder
        self.up4 = UpblockAttention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpblockAttention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpblockAttention(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpblockAttention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        
        # Activation
        self.last_activation = nn.Sigmoid()
        self.multi_activation = nn.Softmax(dim=1)
        
        # Text processing
        self.text_module4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        
        # Load CLIP
        self.load_clip(config)
        
        # Reconstruction components (only used when aux=True)
        self.mlp_woi = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )
        self.mlp_visual = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )
        
        self.rec_text_trans1 = DualTransformer(
            num_heads=4, 
            num_decoder_layers1=self.config.rec_trans_num_layers1, 
            num_decoder_layers2=self.config.rec_trans_num_layers1
        )
        self.rec_img_trans1 = DualTransformer(
            num_heads=4, 
            num_decoder_layers1=self.config.rec_trans_num_layers1, 
            num_decoder_layers2=self.config.rec_trans_num_layers1
        )
        
        self.mse_loss = nn.MSELoss(reduction='none')
        self.loss_fct = CrossEn(config)
        self.dropout1 = nn.Dropout(p=config.dropout_value)
        
        # Auxiliary flag for reconstruction (set externally based on training mode)
        self.aux = True
        
    def load_clip(self, config):
        backbone = config.clip_backbone
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CLIP model not found at {model_path}")
        try:
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
        
        print("Using CLIP version:", model_path)
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")]
        )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        
        self.clip = CLIP(
            embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )
        
        if torch.cuda.is_available():
            convert_weights(self.clip)
        self.clip.load_state_dict(state_dict, strict=False)
        self.clip.float()
        
        if self.config.frozen_clip:
            for param in self.clip.parameters():
                param.requires_grad = False

    def _mask_feat(self, feat, feat_len, weights=None, mask_rate=0.3, mode='dist', mask_idx='1', mask_num=None):
        masked_vec = []
        for i, l in enumerate(feat_len):
            l = int(l)
            if mask_num is not None:
                num_masked_vec = max(int(mask_num), 1)
            else:
                num_masked_vec = max(int(l * mask_rate), 1)
            masked_vec.append(torch.zeros([feat.size(1)]).byte().cuda())
            if l < 1:
                continue
            p = weights[i, :l].cpu().detach().numpy() if weights is not None else None
            
            if mode == 'dist':
                if p is not None:
                    # Ensure probabilities are valid
                    p = np.clip(p, 0, None)
                    p_sum = np.sum(p)
                    if p_sum > 0:
                        p = p / p_sum
                    else:
                        p = None
                    
                    if p is not None and np.sum(p > 0) <= num_masked_vec:
                        num_masked_vec = max(int(np.sum(p > 0)), 1)
                
                num_to_sample = min(num_masked_vec, l)
                choices = np.random.choice(np.arange(l), num_to_sample, replace=False, p=p)
                
            elif mode == 'topk':
                choices = torch.topk(weights[i, :l], k=min(num_masked_vec, l))[1].cpu().numpy()
            else:
                choices = np.random.choice(np.arange(l), min(num_masked_vec, l), replace=False)
                
            masked_vec[-1][choices] = 1

        masked_vec = torch.stack(masked_vec, 0).unsqueeze(-1)
        if mask_idx == '1':
            out_feat = feat.masked_fill(masked_vec == 1, 0)
        elif mask_idx == '0':
            out_feat = feat.masked_fill(masked_vec == 0, 0)
        else:
            out_feat = feat

        return out_feat, masked_vec
    
    def reconstructor(self, text_feat4, text_feat, text_mask, img_feat4, img_feat, text_feat_clip):
        bsz, l, T = text_feat.shape
        img_feat4_flat = img_feat4.flatten(2).transpose(-1, -2)
        img_mask = torch.ones((bsz, img_feat4_flat.shape[1])).to(text_mask.device)
        
        img_feat_inter = img_feat.clone()
        
        if self.config.dropout:
            img_feat_for_weight = self.dropout1(img_feat)
        else:
            img_feat_for_weight = img_feat
            
        img_weight = self.mlp_visual(img_feat_for_weight).squeeze(2)
        img_weight = img_weight.masked_fill_(~img_mask.bool(), float("-inf"))
        img_weight = torch.softmax(img_weight, dim=-1)
        
        masked_img_feat, _ = self._mask_feat(
            img_feat4_flat, img_mask.sum(1), weights=img_weight, 
            mask_rate=self.config.img_mask_rate, mode=self.config.mask_mode
        )

        if self.config.dropout:
            text_feat_for_weight = self.dropout1(text_feat)
        else:
            text_feat_for_weight = text_feat
            
        text_weight = self.mlp_woi(text_feat_for_weight).squeeze(2)
        text_weight = text_weight.masked_fill_(~text_mask.bool(), float("-inf"))
        text_weight = torch.softmax(text_weight, dim=-1)
        
        masked_text_feat, _ = self._mask_feat(
            text_feat4, text_mask.sum(1), weights=text_weight, 
            mask_rate=self.config.text_mask_rate, mode=self.config.mask_mode
        )

        img_rec_res = self.lambda_reconstructor(text_feat4, masked_img_feat, text_weight, img_mask, mode="img")
        text_rec_res = self.lambda_reconstructor(img_feat4_flat, masked_text_feat, img_weight, text_mask, mode="text")

        loss_text_rec = self.mse_loss(text_rec_res, text_feat_clip).mean()
        loss_img_rec = self.mse_loss(img_rec_res, img_feat_inter).mean()

        if torch.isnan(loss_text_rec):
            loss_text_rec = torch.tensor(0.0, device=loss_text_rec.device)
        if torch.isnan(loss_img_rec):
            loss_img_rec = torch.tensor(0.0, device=loss_img_rec.device)

        return {
            "loss_text_rec": loss_text_rec,
            "loss_img_rec": loss_img_rec,
        }, img_weight, text_weight

    def lambda_reconstructor(self, src1, src2, weight, bool_mask, mode="text"):
        if mode == "text":
            rec_res = self.rec_text_trans1(src1, None, src2, None, decoding=2, gauss_weight=weight)[1]
        elif mode == "img":
            rec_res = self.rec_img_trans1(src1, None, src2, None, decoding=2, gauss_weight=weight)[1]
        return rec_res

    def cond_cons_loss(self, text_feat, text_mask, img_feat, text_weight=None, img_weight=None):
        img_feat_flat = img_feat.flatten(2).transpose(-1, -2)
        
        # Use ControlGAN attention weights if available
        if self.use_controlgan and hasattr(self.interact, 'get_attention_weights'):
            controlgan_attn = self.interact.get_attention_weights()
            if controlgan_attn is not None:
                if img_weight is None:
                    img_weight = controlgan_attn.mean(dim=-1)
                if text_weight is None:
                    text_weight = controlgan_attn.mean(dim=1)
        
        # Default weights if not provided
        if text_weight is None:
            text_weight = torch.ones(text_feat.shape[0], text_feat.shape[1], device=text_feat.device)
        if img_weight is None:
            img_weight = torch.ones(img_feat_flat.shape[0], img_feat_flat.shape[1], device=img_feat_flat.device)
            
        text_weight = torch.softmax(text_weight, dim=-1)
        img_weight = torch.softmax(img_weight, dim=-1)

        text_feat_norm = F.normalize(text_feat, dim=-1, p=2)
        img_feat_norm = F.normalize(img_feat_flat, dim=-1, p=2)

        retrieve_logits = torch.einsum('atd,bvd->abtv', [text_feat_norm, img_feat_norm])
        retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, text_mask.float()])

        t2v_logits, _ = retrieve_logits.max(dim=-1)
        t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])
        v2t_logits, _ = retrieve_logits.max(dim=-2)
        v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, img_weight])
        retrieve_logits_final = (t2v_logits + v2t_logits) / 2.0

        logit_scale = self.clip.logit_scale.exp()
        loss_t2v = self.loss_fct(retrieve_logits_final * logit_scale)
        loss_v2t = self.loss_fct(retrieve_logits_final.T * logit_scale)
        
        return (loss_t2v + loss_v2t) / 2

    def forward(self, images, masks, text_token, text_mask, mode="train"):
        loss_dic = {}
        
        # Process text
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        cls, text_feat = self.clip.encode_text(text_token, return_hidden=True, mask=text_mask)
        text_feat_clip = text_feat.clone()
        
        # Encoder
        x = images.float()
        x1 = self.inc(x)
        
        text_feat = self.text_module4(text_feat.float().transpose(1, 2)).transpose(1, 2)
        
        img_feat1 = self.down1(x1)
        img_feat2 = self.down2(img_feat1)
        img_feat3 = self.down3(img_feat2)
        img_feat4 = self.down4(img_feat3)
        
        # ================================================================
        # Interaction - BOTH Interactor and EnhancedInteractor return 2 values
        # ================================================================
        text_feat4_4rec, img_feat4_4rec = self.interact(img_feat4, text_feat, text_mask)
        
        # Initialize weights
        img_weight = None
        text_weight = None
        
        # Reconstruction loss (only if aux is True)
        if self.aux:
            try:
                rec_loss_dic, img_weight, text_weight = self.reconstructor(
                    text_feat, text_feat4_4rec, text_mask, 
                    img_feat4, img_feat4_4rec, text_feat_clip
                )
                loss_dic.update(rec_loss_dic)
                
                # Use .get() for safe access to loss_weight
                loss_weight = self.loss_weight if isinstance(self.loss_weight, dict) else {}
                if loss_weight.get("loss_ccl", 0) != 0:
                    cond_cons_loss = self.cond_cons_loss(
                        text_feat, text_mask, img_feat4, 
                        text_weight=text_weight, img_weight=img_weight
                    )
                    loss_dic["loss_ccl"] = cond_cons_loss
            except Exception as e:
                print(f"Reconstruction error (disabling): {e}")
                self.aux = False
        
        # Decoder
        x = self.up4(img_feat4_4rec.transpose(-1, -2).view(-1, 512, 14, 14), img_feat3)
        x = self.up3(x, img_feat2)
        x = self.up2(x, img_feat1)
        x = self.up1(x, x1)
        
        # Output
        if self.n_classes == 1:
            x = self.outc(x)
            logits = self.last_activation(x)
        else:
            logits = self.outc(x)
        
        if mode == "test":
            return logits, img_weight, text_weight
        else:
            return logits, loss_dic