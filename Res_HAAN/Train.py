

import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
import ast
from tqdm import tqdm
from sklearn.metrics import f1_score, mean_absolute_error,mean_squared_error, classification_report, confusion_matrix, r2_score
from sklearn.model_selection import StratifiedKFold
import timm
import cv2
import json
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random

# ===================================================================
# Part 1: Set a fixed random seed
# ===================================================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"--- Random seed has been fixed to: {seed} ---")

# ===================================================================
# Part 2: Configuration Class
# ===================================================================
class ProjectConfig:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "data")
    RAW_CSV_PATH = os.path.join(DATA_DIR, "Data_RSW.csv")
    FINAL_METADATA_PATH = os.path.join(DATA_DIR, "final_data.csv")
    MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "saved_models")
    
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    STATS_PATH = os.path.join(DATA_DIR, 'regression_stats.json')
    REGRESSION_TARGETS = ['NuggetDiameter (mm)', 'PullTest (N)']
    LABEL_MAPPING = {'Good': 0, 'Bad': 1, 'Explode': 2}
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    EMBED_DIM = 256
    NUM_HEADS = 4
    DROPOUT = 0.2
    BACKBONE = 'resnet18' 
    
    BATCH_SIZE = 16
    LEARNING_RATE_HEAD = 1e-4      
    LEARNING_RATE_BACKBONE = 1e-5  
    WEIGHT_DECAY = 1e-4
    EPOCHS = 150         
    PATIENCE = 25        
    N_SPLITS = 5 
    
    IMG_SIZE = 224
    USE_CONSISTENCY_LOSS = True
    CONSISTENCY_LOSS_WEIGHT = 0.5
    FOCAL_GAMMA = 3.0  #  The gamma value for the focal loss has been made configurable
    
    # ==========================================
    #  Ablation Experiment Switch Master Control
    # ==========================================
    # 1. Mode-selective switch: 'all', 'vision_only', 'temporal_only'
    MODALITY_MODE = 'all' 
    
    # 2. Combined architecture ablation switch: 'haan', 'concat', 'symmetric', 'cmt', 'healnet'
    FUSION_MODE = 'haan' 
    
    # 3. Training strategy ablation switch (used for generating Table 3)
    # Single strategies: 'baseline', 'ros', 'focal', 'ldam', 'cbl'
    # Hybrid strategies: 'smote+ldam', 'smote+cbl', 'proposed'
    TRAINING_STRATEGY = 'proposed'
    
    # Other experimental switches
    REVERSE_QK = False
    TASK_MODE = 'multi' #'cls_only',  'reg_only', 'multi'
    OCCLUSION_TEST = False 
    EVAL_WITH_ARTIFACTS = False
    # Physical A Priori Injection Toggle (specifically for Explode splatter)
    USE_PHYSICAL_PRIOR = True

# ===================================================================
# Part 3: Models & Losses
# ===================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = torch.tensor(alpha, device=ProjectConfig.DEVICE) if alpha is not None else None
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss

#  LDAM Loss (Label-Distribution-Aware Margin Loss)
class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30, weight=None, device='cuda'):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.tensor(m_list, dtype=torch.float32, device=device)
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = F.one_hot(target, num_classes=len(self.m_list)).bool()
        batch_m = self.m_list.unsqueeze(0).expand(x.size(0), -1)
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)

#  CBL (Class-Balanced Weight Calculator)
def get_cb_weights(cls_num_list, beta=0.9999):
    effective_num = 1.0 - np.power(beta, cls_num_list)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(cls_num_list)
    return torch.tensor(weights, dtype=torch.float32)

class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super(MultiTaskLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    def forward(self, cls_loss, reg_loss):
        precision_cls = torch.exp(-self.log_vars[0])
        loss_cls = precision_cls * cls_loss + self.log_vars[0]
        precision_reg = torch.exp(-self.log_vars[1])
        loss_reg = precision_reg * reg_loss + self.log_vars[1]
        return loss_cls + loss_reg

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class AdvancedImageFusionEncoder(nn.Module):
    def __init__(self, cfg: ProjectConfig):
        super().__init__()
        self.image_encoder = timm.create_model(cfg.BACKBONE, pretrained=True, num_classes=cfg.EMBED_DIM)
        self.ir_cross_attn = CrossAttention(cfg.EMBED_DIM, cfg.NUM_HEADS, cfg.DROPOUT)
        self.front_cross_attn = CrossAttention(cfg.EMBED_DIM, cfg.NUM_HEADS, cfg.DROPOUT)
        self.back_cross_attn = CrossAttention(cfg.EMBED_DIM, cfg.NUM_HEADS, cfg.DROPOUT)
        self.fusion_layer = nn.Sequential(nn.Linear(cfg.EMBED_DIM * 3, cfg.EMBED_DIM), nn.ReLU(), nn.LayerNorm(cfg.EMBED_DIM))
        
    def forward(self, ir, front, back):
        if self.training and random.random() < 0.1:
            drop_idx = random.randint(0, 2)
            if drop_idx == 0: ir = torch.zeros_like(ir)
            elif drop_idx == 1: front = torch.zeros_like(front)
            else: back = torch.zeros_like(back)

        ir_feat = self.image_encoder(ir).unsqueeze(1)
        front_feat = self.image_encoder(front).unsqueeze(1)
        back_feat = self.image_encoder(back).unsqueeze(1)
        
        ir_fused = self.ir_cross_attn(ir_feat, torch.cat([front_feat, back_feat], dim=1))
        front_fused = self.front_cross_attn(front_feat, torch.cat([ir_feat, back_feat], dim=1))
        back_fused = self.back_cross_attn(back_feat, torch.cat([ir_feat, front_feat], dim=1))
        
        combined = torch.cat([ir_fused, front_fused, back_fused], dim=-1).squeeze(1)
        return self.fusion_layer(combined)

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    def forward(self, query, context):
        attn_output, _ = self.attn(query, context, context)
        return self.norm(query + attn_output)

class Lightweight_HAAN_Net(nn.Module):
    def __init__(self, param_dim, cfg: ProjectConfig):
        super().__init__()
        self.cfg = cfg
        self.image_fusion_encoder = AdvancedImageFusionEncoder(cfg)
        
        lstm_hidden_dim = 128
        self.lstm = nn.LSTM(input_size=param_dim, hidden_size=lstm_hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.temporal_proj = nn.Linear(lstm_hidden_dim * 2, cfg.EMBED_DIM)
        self.pos_encoder = PositionalEncoding(cfg.EMBED_DIM, cfg.DROPOUT)
        
        fused_dim = cfg.EMBED_DIM * 2
        
        # Architecture Module Definition
        if cfg.MODALITY_MODE in ['vision_only', 'temporal_only']:
            fused_dim = cfg.EMBED_DIM
        else:
            if cfg.FUSION_MODE == 'haan':
                self.q_proj = nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM)
                self.k_proj = nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM)
                self.v_proj = nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM)
                self.cross_attention = nn.MultiheadAttention(cfg.EMBED_DIM, cfg.NUM_HEADS, batch_first=True)
                self.fusion_norm = nn.LayerNorm(cfg.EMBED_DIM)
                fused_dim = cfg.EMBED_DIM * 2
                
            elif cfg.FUSION_MODE == 'symmetric':
                self.self_attention = nn.MultiheadAttention(cfg.EMBED_DIM, cfg.NUM_HEADS, batch_first=True)
                self.fusion_norm = nn.LayerNorm(cfg.EMBED_DIM)
                fused_dim = cfg.EMBED_DIM
                
            elif cfg.FUSION_MODE == 'concat':
                fused_dim = cfg.EMBED_DIM * 2
                
            elif cfg.FUSION_MODE == 'cmt': 
                # [SOTA 1] CMT Shuangliu Divided Attention
                self.cmt_attn_img = nn.MultiheadAttention(cfg.EMBED_DIM, cfg.NUM_HEADS, batch_first=True)
                self.cmt_attn_tmp = nn.MultiheadAttention(cfg.EMBED_DIM, cfg.NUM_HEADS, batch_first=True)
                self.fusion_norm = nn.LayerNorm(cfg.EMBED_DIM * 2)
                fused_dim = cfg.EMBED_DIM * 2
                
            elif cfg.FUSION_MODE == 'healnet':
                # [SOTA 2] HEALNet Shared Subspace
                self.num_latents = 8 
                self.latent_tokens = nn.Parameter(torch.randn(1, self.num_latents, cfg.EMBED_DIM))
                self.heal_cross_attn = nn.MultiheadAttention(cfg.EMBED_DIM, cfg.NUM_HEADS, batch_first=True)
                self.heal_self_attn = nn.MultiheadAttention(cfg.EMBED_DIM, cfg.NUM_HEADS, batch_first=True)
                self.fusion_norm = nn.LayerNorm(cfg.EMBED_DIM)
                fused_dim = cfg.EMBED_DIM

        #if cfg.TASK_MODE in ['multi', 'cls_only']:
        #    self.classifier = nn.Sequential(nn.Linear(fused_dim, cfg.EMBED_DIM), nn.ReLU(), nn.Dropout(cfg.DROPOUT), nn.Linear(cfg.EMBED_DIM, len(cfg.LABEL_MAPPING)))

        if cfg.TASK_MODE in ['multi', 'cls_only']:
            # If physical prior is enabled, an additional temporal difference dimension (param_dim) and a visual infrared extreme value dimension (1) are added.
            cls_in_dim = fused_dim + param_dim + 1 if cfg.USE_PHYSICAL_PRIOR else fused_dim
            self.classifier = nn.Sequential(
                nn.Linear(cls_in_dim, cfg.EMBED_DIM), 
                nn.ReLU(), 
                nn.Dropout(cfg.DROPOUT), 
                nn.Linear(cfg.EMBED_DIM, len(cfg.LABEL_MAPPING))
            )
        if cfg.TASK_MODE in ['multi', 'reg_only']:
            self.regressor = nn.Sequential(nn.Linear(fused_dim, cfg.EMBED_DIM), nn.ReLU(), nn.Dropout(cfg.DROPOUT), nn.Linear(cfg.EMBED_DIM, len(cfg.REGRESSION_TARGETS)))
            
    def forward(self, ir_image, rgb_front_image, rgb_back_image, params_list, class_labels=None, reg_labels=None):
        image_feature = self.image_fusion_encoder(ir_image, rgb_front_image, rgb_back_image)
        lstm_out, _ = self.lstm(params_list)
        temporal_feats = self.pos_encoder(self.temporal_proj(lstm_out))
        temporal_pool = temporal_feats.mean(dim=1) 

        # --- Forward Propagation: Modality and Architecture Branches ---
        if self.cfg.MODALITY_MODE == 'vision_only': 
            final_feature = image_feature
        elif self.cfg.MODALITY_MODE == 'temporal_only': 
            final_feature = temporal_pool
        else:
            if self.cfg.FUSION_MODE == 'haan':
                if not self.cfg.REVERSE_QK:
                    q = self.q_proj(image_feature.unsqueeze(1))
                    k = self.k_proj(temporal_feats); v = self.v_proj(temporal_feats)
                    attn_out, _ = self.cross_attention(query=q, key=k, value=v)
                    fused_temporal = self.fusion_norm(attn_out.squeeze(1))
                else:
                    img_kv = image_feature.unsqueeze(1).expand(-1, temporal_feats.size(1), -1)
                    q = self.q_proj(temporal_feats)
                    k = self.k_proj(img_kv); v = self.v_proj(img_kv)
                    attn_out, _ = self.cross_attention(query=q, key=k, value=v)
                    fused_temporal = self.fusion_norm(attn_out.mean(dim=1))
                final_feature = torch.cat([image_feature, fused_temporal], dim=1)
                
            elif self.cfg.FUSION_MODE == 'symmetric':
                tokens = torch.stack([image_feature, temporal_pool], dim=1)
                attn_out, _ = self.self_attention(query=tokens, key=tokens, value=tokens)
                final_feature = self.fusion_norm(attn_out.mean(dim=1))
                
            elif self.cfg.FUSION_MODE == 'concat':
                final_feature = torch.cat([image_feature, temporal_pool], dim=1)
                
            elif self.cfg.FUSION_MODE == 'cmt': 
                img_seq = image_feature.unsqueeze(1)
                attn_img, _ = self.cmt_attn_img(query=img_seq, key=temporal_feats, value=temporal_feats)
                img_kv = img_seq.expand(-1, temporal_feats.size(1), -1)
                attn_tmp, _ = self.cmt_attn_tmp(query=temporal_feats, key=img_kv, value=img_kv)
                final_feature = self.fusion_norm(torch.cat([attn_img.squeeze(1), attn_tmp.mean(dim=1)], dim=1))
                
            elif self.cfg.FUSION_MODE == 'healnet':
                bs = image_feature.size(0)
                latents = self.latent_tokens.expand(bs, -1, -1)
                all_modality_seq = torch.cat([image_feature.unsqueeze(1), temporal_feats], dim=1)
                latents, _ = self.heal_cross_attn(query=latents, key=all_modality_seq, value=all_modality_seq)
                latents, _ = self.heal_self_attn(query=latents, key=latents, value=latents)
                final_feature = self.fusion_norm(latents.mean(dim=1))

        # ========================================================
        # Feature-level SMOTE (Triggered intelligently by policy switches, with absolute isolation of physical artefacts)
        # ========================================================
        if self.training and class_labels is not None and reg_labels is not None:
            # Only these advanced strategies allow feature layer interpolation to be triggered
            if self.cfg.TRAINING_STRATEGY in ['smote+ldam', 'smote+cbl', 'proposed', 'smote']:
                for cls in [1, 2]:
                    cls_mask = class_labels == cls
                    if cls_mask.sum() > 1:
                        feats = final_feature[cls_mask]
                        regs = reg_labels[cls_mask]
                        rand_idx = torch.randperm(feats.size(0))
                        neighbor_feats = feats[rand_idx]
                        neighbor_regs = regs[rand_idx]
                        
                        # Limit the alpha value to a maximum of 0.2 to prevent interpolation from crossing the decision boundary and contaminating the good samples.
                        alpha = (torch.rand(feats.size(0), 1) * 0.2).to(final_feature.device)
                        
                        syn_feats = feats + alpha * (neighbor_feats - feats)
                        syn_regs = regs + alpha * (neighbor_regs - regs)
                        syn_labels = torch.full((feats.size(0),), cls, dtype=class_labels.dtype, device=final_feature.device)
                        
                        final_feature = torch.cat([final_feature, syn_feats], dim=0)
                        class_labels = torch.cat([class_labels, syn_labels], dim=0)
                        reg_labels = torch.cat([reg_labels, syn_regs], dim=0)

# ========================================================
        # (Physical Prior Injection) - Specialised in addressing missed detection of splashes
        # ========================================================
        if self.cfg.USE_PHYSICAL_PRIOR:
            # 1. Extracting temporal transient features (calculating the maximum absolute value of the first-order difference to capture instantaneous drops in resistance or current)
            if params_list.size(1) > 1:
                # Shape: [batch, seq_len-1, param_dim]
                temporal_diff = torch.abs(params_list[:, 1:, :] - params_list[:, :-1, :])
                # Select the value with the most significant mutation in the entire sequence : [batch, param_dim]
                mutation_prior, _ = torch.max(temporal_diff, dim=1) 
                # [New] Add this line to cut off the gradient!
                mutation_prior = mutation_prior.detach()                
            else:
                mutation_prior = torch.zeros(final_feature.size(0), params_list.size(-1), device=final_feature.device)
            
            # 2. Extracting visual splash features (splashes cause abnormal bright spots in infrared images)
            # ir_image shape: [batch, 3, H, W]
            ir_flatten = ir_image.view(ir_image.size(0), -1)
            # Extract the brightest pixel value from each image : [batch, 1]
            visual_spike_prior, _ = torch.max(ir_flatten, dim=1, keepdim=True)
            # [New] Add this line to cut off the gradient!
            visual_spike_prior = visual_spike_prior.detach()
            
            # If we are in training mode and SMOTE has been triggered, we need to add fabricated prior features to the generated synthetic samples as well.
            if self.training and class_labels is not None and final_feature.size(0) > mutation_prior.size(0):
                num_syn = final_feature.size(0) - mutation_prior.size(0)
                # 简单用均值填充虚拟样本的先验特征，防止维度不匹配
                syn_mutation = mutation_prior.mean(dim=0, keepdim=True).expand(num_syn, -1)
                syn_visual = visual_spike_prior.mean(dim=0, keepdim=True).expand(num_syn, -1)
                mutation_prior = torch.cat([mutation_prior, syn_mutation], dim=0)
                visual_spike_prior = torch.cat([visual_spike_prior, syn_visual], dim=0)

            # Concatenate pure ‘physical mutation features’ with deep features, and feed them directly into the classifier!
            classifier_input = torch.cat([final_feature, mutation_prior, visual_spike_prior], dim=1)
        else:
            classifier_input = final_feature


        cls_out = self.classifier(classifier_input) if hasattr(self, 'classifier') else None
        reg_out = self.regressor(final_feature) if hasattr(self, 'regressor') else None
        
        if self.training and class_labels is not None:
            return cls_out, reg_out, class_labels, reg_labels
        return cls_out, reg_out

# ===================================================================
# Part 4: Dataloader
# ===================================================================
def prepare_all_data(cfg):
    if os.path.exists(cfg.FINAL_METADATA_PATH) and os.path.exists(cfg.STATS_PATH):
        print("--- STEP 1: Metadata If it exists, load it directly.---")
        with open(cfg.STATS_PATH, 'r') as f: stats = json.load(f)
        df_final = pd.read_csv(cfg.FINAL_METADATA_PATH)
        param_cols = [c for c in df_final.columns if c not in ['ID', 'ir_filename', 'rgb_f_filename', 'rgb_b_filename', 'class_label_id', 'regression_label_normalized', 'regression_label_raw']]
        return param_cols, stats
    else:
        raise FileNotFoundError(f"No pre-processed data found {cfg.FINAL_METADATA_PATH}. Please run the original pre-processing script to generate the output.")

def get_train_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler

def custom_collate_fn(batch):
    keys_to_stack = [k for k in batch[0].keys() if k != 'params_list']
    collated = {key: torch.stack([d[key] for d in batch]) for key in keys_to_stack}
    params_lists = [d['params_list'] for d in batch]
    if not all(p.nelement() > 0 for p in params_lists): 
        collated['params_list'] = torch.zeros(len(batch), 0, 0)
        return collated
    max_len = max(p.shape[0] for p in params_lists)
    if params_lists and params_lists[0].nelement() > 0:
        param_dim = params_lists[0].shape[1]
        padded_params = torch.zeros(len(batch), max_len, param_dim)
        for i, p in enumerate(params_lists): padded_params[i, :p.shape[0], :] = p
        collated['params_list'] = padded_params
    else: collated['params_list'] = torch.zeros(len(batch), 0, 0)
    return collated

class WeldingDataset(Dataset):
    def __init__(self, metadata, cfg: ProjectConfig, param_columns, stats, is_train=False):
        self.cfg = cfg
        self.metadata = metadata.reset_index(drop=True)
        self.param_columns = param_columns
        self.is_train = is_train
        
        for col in self.param_columns + ['regression_label_normalized', 'regression_label_raw']:
            if isinstance(self.metadata[col].iloc[0], str):
                self.metadata[col] = self.metadata[col].apply(ast.literal_eval)

        self.transform_base = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)), 
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.transform_aug = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)), 
            transforms.RandomHorizontalFlip(), transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)), 
            transforms.ColorJitter(brightness=0.2, contrast=0.2),      
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def process_image(self, img_path, transform):
        full_path = os.path.join(self.cfg.DATA_DIR, img_path)
        img = cv2.imread(full_path)
        if img is None: return torch.zeros(3, self.cfg.IMG_SIZE, self.cfg.IMG_SIZE)
        img_tensor = transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if self.cfg.OCCLUSION_TEST and not self.is_train:
            c, h, w = img_tensor.shape
            img_tensor[:, h//2-25:h//2+25, w//2-25:w//2+25] = 0
        return img_tensor

    def __len__(self): return len(self.metadata)
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        params_list = [row[col] for col in self.param_columns]
        params_array = np.array(params_list).T.astype(np.float32)
        if params_array.size > 0:
            mean, std = np.mean(params_array, axis=0), np.std(params_array, axis=0)
            params_array = (params_array - mean) / (std + 1e-6)
            
        eval_transform = self.transform_aug if (not self.is_train and self.cfg.EVAL_WITH_ARTIFACTS) else self.transform_base
        
        return {
            "ir_image": self.process_image(row['ir_filename'], eval_transform), 
            "rgb_front_image": self.process_image(row['rgb_f_filename'], eval_transform), 
            "rgb_back_image": self.process_image(row['rgb_b_filename'], eval_transform), 
            "ir_image_aug": self.process_image(row['ir_filename'], self.transform_aug) if self.is_train else torch.zeros(3, 224, 224), 
            "rgb_front_image_aug": self.process_image(row['rgb_f_filename'], self.transform_aug) if self.is_train else torch.zeros(3, 224, 224), 
            "rgb_back_image_aug": self.process_image(row['rgb_b_filename'], self.transform_aug) if self.is_train else torch.zeros(3, 224, 224), 
            "params_list": torch.tensor(params_array), 
            "class_label": torch.tensor(row['class_label_id'], dtype=torch.long), 
            "reg_label": torch.tensor(row['regression_label_normalized'], dtype=torch.float32), 
            "reg_label_raw": torch.tensor(row['regression_label_raw'], dtype=torch.float32)
        }

# ===================================================================
# Part 5: K-Fold + Early Stopping Training Loop
# ===================================================================
def run_kfold_training(cfg, df_final, param_columns, stats):
    print(f"\n=== Start  {cfg.N_SPLITS}-Fold stratified cross-validation ===")
    print(f"[Configuration] Modality: {cfg.MODALITY_MODE} | Fusion: {cfg.FUSION_MODE} | Strategy: {cfg.TRAINING_STRATEGY}")
    
    skf = StratifiedKFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=42)
    fold_metrics = {'f1': [], 'mae_nugget': [], 'mae_pull': [], 'r2_nugget': [], 'r2_pull': [], 
                    'rmse_nugget': [], 'rmse_pull': [], 'nrmse_nugget': [], 'nrmse_pull': []}

    all_fold_labels, all_fold_preds = [], []
    target_names = list(cfg.LABEL_MAPPING.keys()) 
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_final, df_final['class_label_id'])):
        print(f"\n--- Fold {fold+1}/{cfg.N_SPLITS} ---")
        
        train_df = df_final.iloc[train_idx]
        val_df = df_final.iloc[val_idx]
        train_labels = train_df['class_label_id'].values
        
        # Calculate the category distribution for this fold (for LDAM and CBL)
        cls_counts = np.bincount(train_labels)
        
        # Allocation Strategy 1: DataLoader Sampling Strategy
        if cfg.TRAINING_STRATEGY in ['ros', 'smote', 'smote+ldam', 'smote+cbl', 'proposed']:
            sampler = get_train_sampler(train_labels)
            train_loader = DataLoader(WeldingDataset(train_df, cfg, param_columns, stats, is_train=True), batch_size=cfg.BATCH_SIZE, sampler=sampler, collate_fn=custom_collate_fn)
        else:
            # baseline, focal, ldam, cbl 使用标准 shuffle
            train_loader = DataLoader(WeldingDataset(train_df, cfg, param_columns, stats, is_train=True), batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
            
        val_loader = DataLoader(WeldingDataset(val_df, cfg, param_columns, stats, is_train=False), batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
        
        model = Lightweight_HAAN_Net(len(param_columns), cfg).to(cfg.DEVICE)

        # Allocation Strategy 2: Classification Loss Function
        if cfg.TRAINING_STRATEGY in ['focal', 'proposed']:
            class_criterion = FocalLoss(alpha=[0.3, 0.35, 0.35], gamma=cfg.FOCAL_GAMMA).to(cfg.DEVICE)
        elif cfg.TRAINING_STRATEGY in ['ldam', 'smote+ldam']:
            class_criterion = LDAMLoss(cls_num_list=cls_counts, device=cfg.DEVICE).to(cfg.DEVICE)
        elif cfg.TRAINING_STRATEGY in ['cbl', 'smote+cbl']:
            cb_weights = get_cb_weights(cls_counts).to(cfg.DEVICE)
            class_criterion = nn.CrossEntropyLoss(weight=cb_weights).to(cfg.DEVICE)
        else:
            # baseline, ros, smote
            class_criterion = nn.CrossEntropyLoss().to(cfg.DEVICE)
            
        reg_criterion = nn.SmoothL1Loss(beta=1.0)
        mtl_loss = MultiTaskLoss(num_tasks=2).to(cfg.DEVICE)
        
        backbone_params = list(model.image_fusion_encoder.image_encoder.parameters())
        head_params = [p for n, p in model.named_parameters() if 'image_encoder' not in n]
        optimizer = optim.AdamW([{'params': backbone_params, 'lr': cfg.LEARNING_RATE_BACKBONE}, {'params': head_params, 'lr': cfg.LEARNING_RATE_HEAD}, {'params': mtl_loss.parameters(), 'lr': cfg.LEARNING_RATE_HEAD}], weight_decay=cfg.WEIGHT_DECAY)
        
        scaler = GradScaler('cuda') 
        
        best_score = -float('inf') 
        best_f1, best_mae_nugget, best_mae_pull = 0, float('inf'), float('inf')
        best_r2_nugget, best_r2_pull = -float('inf'), -float('inf') 
        best_epoch, epochs_no_improve = 0, 0
        best_rmse_nugget, best_rmse_pull = float('inf'), float('inf')
        best_nrmse_nugget, best_nrmse_pull = float('inf'), float('inf')
        best_cls_report, best_val_labels, best_val_preds = "", [], []
        
        for epoch in range(cfg.EPOCHS):
            model.train()
            for batch in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{cfg.EPOCHS}", leave=False):
                optimizer.zero_grad(set_to_none=True)
                inputs = {
                    "ir_image": batch["ir_image"].to(cfg.DEVICE), "rgb_front_image": batch["rgb_front_image"].to(cfg.DEVICE), 
                    "rgb_back_image": batch["rgb_back_image"].to(cfg.DEVICE), "params_list": batch["params_list"].to(cfg.DEVICE)
                }
                labels = batch['class_label'].to(cfg.DEVICE)
                reg_targets = batch['reg_label'].to(cfg.DEVICE)
                
                with autocast('cuda', dtype=torch.float16):
                    cls_preds, reg_preds, syn_labels, syn_reg_targets = model(**inputs, class_labels=labels, reg_labels=reg_targets)
                    c_loss = class_criterion(cls_preds, syn_labels) if cfg.TASK_MODE in ['multi', 'cls_only'] else 0
                    r_loss = reg_criterion(reg_preds, syn_reg_targets) if cfg.TASK_MODE in ['multi', 'reg_only'] else 0
                    main_loss = mtl_loss(c_loss, r_loss) if cfg.TASK_MODE == 'multi' else (c_loss if cfg.TASK_MODE == 'cls_only' else r_loss)
                    
                    if cfg.USE_CONSISTENCY_LOSS:
                        inputs_aug = { "ir_image": batch["ir_image_aug"].to(cfg.DEVICE), "rgb_front_image": batch["rgb_front_image_aug"].to(cfg.DEVICE), "rgb_back_image": batch["rgb_back_image_aug"].to(cfg.DEVICE), "params_list": batch["params_list"].to(cfg.DEVICE)}
                        cls_preds_aug, reg_preds_aug = model(**inputs_aug)
                        orig_bs = labels.size(0)
                        consist_loss = 0
                        if cfg.TASK_MODE in ['multi', 'cls_only']: consist_loss += F.mse_loss(F.softmax(cls_preds_aug, dim=1), F.softmax(cls_preds[:orig_bs].detach(), dim=1))
                        if cfg.TASK_MODE in ['multi', 'reg_only']: consist_loss += reg_criterion(reg_preds_aug, reg_preds[:orig_bs].detach())
                        main_loss += cfg.CONSISTENCY_LOSS_WEIGHT * consist_loss
                
                scaler.scale(main_loss).backward(); scaler.step(optimizer); scaler.update()
            
            model.eval()
            val_preds, val_labels, val_reg_preds, val_reg_targets = [], [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    inputs = { "ir_image": batch["ir_image"].to(cfg.DEVICE), "rgb_front_image": batch["rgb_front_image"].to(cfg.DEVICE), "rgb_back_image": batch["rgb_back_image"].to(cfg.DEVICE), "params_list": batch["params_list"].to(cfg.DEVICE)}
                    cls_preds, reg_preds = model(**inputs)
                    if cls_preds is not None:
                        val_preds.extend(torch.argmax(cls_preds, 1).cpu().numpy()); val_labels.extend(batch['class_label'].numpy())
                    if reg_preds is not None:
                        reg_preds_unnorm = reg_preds.clone()
                        for i, target_name in enumerate(cfg.REGRESSION_TARGETS):
                            reg_preds_unnorm[:, i] = reg_preds[:, i] * stats[target_name]['std'] + stats[target_name]['mean']
                        val_reg_preds.extend(reg_preds_unnorm.cpu().numpy()); val_reg_targets.extend(batch['reg_label_raw'].numpy())
            
            val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0) if val_preds else 0
            mae_nugget = mean_absolute_error(np.array(val_reg_targets)[:, 0], np.array(val_reg_preds)[:, 0]) if val_reg_preds else 0
            mae_pull = mean_absolute_error(np.array(val_reg_targets)[:, 1], np.array(val_reg_preds)[:, 1]) if val_reg_preds else 0
            r2_nugget = r2_score(np.array(val_reg_targets)[:, 0], np.array(val_reg_preds)[:, 0]) if val_reg_preds else 0
            r2_pull = r2_score(np.array(val_reg_targets)[:, 1], np.array(val_reg_preds)[:, 1]) if val_reg_preds else 0

            if val_reg_preds:
                val_reg_targets_np = np.array(val_reg_targets)
                val_reg_preds_np = np.array(val_reg_preds)
                rmse_nugget = np.sqrt(mean_squared_error(val_reg_targets_np[:, 0], val_reg_preds_np[:, 0]))
                rmse_pull = np.sqrt(mean_squared_error(val_reg_targets_np[:, 1], val_reg_preds_np[:, 1]))
                
                range_nugget = np.max(val_reg_targets_np[:, 0]) - np.min(val_reg_targets_np[:, 0])
                nrmse_nugget = rmse_nugget / (range_nugget + 1e-6)
                
                range_pull = np.max(val_reg_targets_np[:, 1]) - np.min(val_reg_targets_np[:, 1])
                nrmse_pull = rmse_pull / (range_pull + 1e-6)
            else:
                rmse_nugget, rmse_pull, nrmse_nugget, nrmse_pull = 0, 0, 0, 0

            # Logic of dynamic evaluation criteria
            if cfg.TASK_MODE in ['multi', 'cls_only']:
                current_score = val_f1
            else:
                # If it's a pure regression task, use negative error as the score (the smaller the error, the higher the score)
                current_score = - (mae_nugget + mae_pull / 1000.0)

            if current_score > best_score:
                best_score = current_score
                best_f1, best_mae_nugget, best_mae_pull = val_f1, mae_nugget, mae_pull
                best_r2_nugget, best_r2_pull = r2_nugget, r2_pull
                best_rmse_nugget, best_rmse_pull = rmse_nugget, rmse_pull
                best_nrmse_nugget, best_nrmse_pull = nrmse_nugget, nrmse_pull
                best_epoch = epoch + 1
                
                best_cls_report = classification_report(val_labels, val_preds, target_names=target_names, zero_division=0) if val_preds else "【纯回归任务 (reg_only)，无分类报告】"
                best_val_labels, best_val_preds = val_labels.copy(), val_preds.copy()
                epochs_no_improve = 0 
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= cfg.PATIENCE:
                print(f"\n[Early Stopping] Fold {fold+1} Stop training at epoch {epoch+1}(no improvement for {cfg.PATIENCE} consecutive epochs)")
                break 
                
        fold_metrics['f1'].append(best_f1); fold_metrics['mae_nugget'].append(best_mae_nugget); fold_metrics['mae_pull'].append(best_mae_pull)
        fold_metrics['r2_nugget'].append(best_r2_nugget); fold_metrics['r2_pull'].append(best_r2_pull)  
        fold_metrics['rmse_nugget'].append(best_rmse_nugget)
        fold_metrics['rmse_pull'].append(best_rmse_pull)
        fold_metrics['nrmse_nugget'].append(best_nrmse_nugget)
        fold_metrics['nrmse_pull'].append(best_nrmse_pull)    
        all_fold_labels.extend(best_val_labels); all_fold_preds.extend(best_val_preds)
        print(f"\nFold {fold+1} Best results ( Epoch {best_epoch}):\n-> F1: {best_f1:.4f} | MAE Ngt: {best_mae_nugget:.2f} (R2: {best_r2_nugget:.2f}) | MAE Pull: {best_mae_pull:.2f} (R2: {best_r2_pull:.2f})")
        print("--- Detailed Categorised Report ---\n", best_cls_report)
        
    print("\n=============================================")
    print(f"--- 5-Fold Summary of final cross-validation results ---")
    print(f"Configuration: Strategy={cfg.TRAINING_STRATEGY}, Modality={cfg.MODALITY_MODE}, Fusion={cfg.FUSION_MODE}")
    if cfg.TASK_MODE in ['multi', 'cls_only']:
        print(f"Macro F1-Score: {np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}")
        cm = confusion_matrix(all_fold_labels, all_fold_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Aggregated 5-Fold CM\n(Strategy: {cfg.TRAINING_STRATEGY}, Fusion: {cfg.FUSION_MODE})')
        plt.ylabel('True Label'); plt.xlabel('Predicted Label')
        save_path = os.path.join(cfg.RESULTS_DIR, f'cm_strat_{cfg.TRAINING_STRATEGY}_fus_{cfg.FUSION_MODE}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=600); plt.close()
        print(f"-> [Success] Aggregated confusion matrix saved to: {save_path}")
        
    if cfg.TASK_MODE in ['multi', 'reg_only']:
        print(f"MAE (Nugget):   {np.mean(fold_metrics['mae_nugget']):.4f} ± {np.std(fold_metrics['mae_nugget']):.4f}")
        print(f"R2 (Nugget):    {np.mean(fold_metrics['r2_nugget']):.4f} ± {np.std(fold_metrics['r2_nugget']):.4f}") 
        print(f"MAE (Pull):     {np.mean(fold_metrics['mae_pull']):.4f} ± {np.std(fold_metrics['mae_pull']):.4f}")
        print(f"R2 (Pull):      {np.mean(fold_metrics['r2_pull']):.4f} ± {np.std(fold_metrics['r2_pull']):.4f}") 
        print(f"\n--- [Internal Test Metrics: RMSE & NRMSE] ---")
        print(f"RMSE  (Nugget): {np.mean(fold_metrics['rmse_nugget']):.4f} ± {np.std(fold_metrics['rmse_nugget']):.4f}")
        print(f"NRMSE (Nugget): {np.mean(fold_metrics['nrmse_nugget'])*100:.2f}% ± {np.std(fold_metrics['nrmse_nugget'])*100:.2f}%")
        print(f"RMSE  (Pull):   {np.mean(fold_metrics['rmse_pull']):.4f} ± {np.std(fold_metrics['rmse_pull']):.4f}")
        print(f"NRMSE (Pull):   {np.mean(fold_metrics['nrmse_pull'])*100:.2f}% ± {np.std(fold_metrics['nrmse_pull'])*100:.2f}%")
        print(f"------------------------------------")
    print("=============================================")

if __name__ == '__main__':
    set_seed(42)

    cfg = ProjectConfig()
    import os

    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    # ==========================================

    # ==========================================
    cfg_base = ProjectConfig()
    
    # ==========================================
    #  Loading public data
    # ==========================================
    try:
        param_columns, stats = prepare_all_data(cfg_base)
        df_final = pd.read_csv(cfg_base.FINAL_METADATA_PATH)
    except FileNotFoundError as e:
        print(e)
        exit()

    # ==========================================
    # Experiment  1: Focal Loss Gamma Sensitivity analysis  
    # ==========================================
    print("\n\n" + "="*50)
    print("Start Experiment 1: Focal Loss Gamma Sensitivity Analysis ")
    print("="*50)
    

    gamma_list = [4.0, 5.0] 
    
    for g in gamma_list:
        set_seed(42) 
        cfg = ProjectConfig()
        cfg.FOCAL_GAMMA = g               
        cfg.CONSISTENCY_LOSS_WEIGHT = 0.5 
        cfg.OCCLUSION_TEST = False        
        print(f"\n>>> Testing Gamma = {g} <<<")
        run_kfold_training(cfg, df_final, param_columns, stats)

    # ==========================================
    #  Experiment 2: Consistency Loss Weight Sensitivity Analysis (Continue)
    # ==========================================
    print("\n\n" + "="*50)
    print("Start Experiment 2: Consistency Loss Weight Sensitivity Analysis")
    print("="*50)
    
    lambda_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    for lam in lambda_list:
        set_seed(42)
        cfg = ProjectConfig()
        cfg.FOCAL_GAMMA = 3.0             
        cfg.CONSISTENCY_LOSS_WEIGHT = lam 
        cfg.OCCLUSION_TEST = False        
        print(f"\n>>> Testing Lambda_inv = {lam} <<<")
        run_kfold_training(cfg, df_final, param_columns, stats)