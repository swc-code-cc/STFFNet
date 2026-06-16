from dataclasses import dataclass
from typing import Callable, Literal

import lightning as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import torch.fft
from lightning import seed_everything
from lightning.pytorch.loggers import WandbLogger
from PIL import Image
from sklearn import metrics as M
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import CatMetric

from src import metrics, plots
from src.config import Backbone, Config, Head
from src.dataset.base import BaseDataset
from src.heads import head
from src.loss import Loss, LossInputs, LossOutputs
from src.losses import unifalign
from src.utils import logger
from scipy.fftpack import dct, idct

class SGFP(nn.Module):
    def __init__(self, drop_ratio=0.2, method='mask', mode='feature+grad', eps=1e-5):
        super().__init__()
        self.drop_ratio = drop_ratio
        self.method = method
        self.mode = mode
        self.eps = eps

    def get_saliency(self, features, logits=None, labels=None):
        if self.mode == 'feature':
            saliency = features.abs()
        elif self.mode == 'feature+grad' and logits is not None and labels is not None:
            grads = torch.autograd.grad(outputs=logits, inputs=features,
                                        grad_outputs=torch.ones_like(logits),
                                        create_graph=True, retain_graph=True)[0]
            saliency = (features.abs() + grads.abs()) / 2
        else:
            saliency = features.abs()
        return saliency

    def forward(self, features, logits=None, labels=None):
        with torch.no_grad():
            saliency = self.get_saliency(features, logits, labels)
            B, D = saliency.shape
            K = max(1, int(self.drop_ratio * D))
            mask = torch.zeros_like(saliency)
            topk = torch.topk(saliency, K, dim=1).indices  # [B, K]
            mask.scatter_(1, topk, 1)

        if self.method == 'mask':
            perturbed = features * (1 - mask)

        elif self.method == 'noise':
            noise = torch.randn_like(features) * self.eps
            perturbed = features + noise * mask

        elif self.method == 'mixup':
            idx = torch.randperm(B)
            perturbed = features * (1 - mask) + features[idx] * mask

        elif self.method == 'invert':
            perturbed = features * (1 - mask) + (-features) * mask

        elif self.method == 'shuffle':
            shuffled = features.clone()
            for i in range(B):
                idx_mask = mask[i].bool()
                if idx_mask.sum() > 1:
                    permuted = features[i, idx_mask][torch.randperm(idx_mask.sum())]
                    shuffled[i, idx_mask] = permuted
            perturbed = shuffled

        elif self.method == 'zero_mean_noise':
            noise = torch.randn_like(features) * self.eps
            noise = noise - noise.mean(dim=1, keepdim=True)
            perturbed = features + noise * mask

        else:
            raise NotImplementedError(f"Unknown ASP method: {self.method}")

        return perturbed, mask



class TCET(nn.Module):
    def __init__(self, dim, nhead=8, num_layers=2, lambda_contrast=0.05):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.norm = nn.LayerNorm(dim)
        self.lambda_contrast = lambda_contrast

    def forward(self, features, video_ids):
        unique_videos = video_ids.unique()
        enhanced_features = features.clone()
        total_loss_contrast = 0

        for vid in unique_videos:
            idxs = (video_ids == vid).nonzero(as_tuple=True)[0]
            seq = features[idxs]  # [T, D]

            if seq.size(0) < 2:
                continue

            cls_tokens = self.cls_token.expand(seq.size(0), -1, -1)
            seq_with_cls = torch.cat([cls_tokens, seq.unsqueeze(0)], dim=1)  # [1, T+1, D]
            trans_feat = self.transformer(seq_with_cls)[0, 1:, :]  # ignore cls token for features

            enhanced_features[idxs] = self.norm(seq + trans_feat)  # residual connection

            # Temporal contrastive loss
            total_loss_contrast += self.temporal_contrastive_loss(trans_feat)

        avg_loss_contrast = total_loss_contrast / len(unique_videos)
        total_loss = self.lambda_contrast * avg_loss_contrast

        return total_loss, enhanced_features

    def temporal_contrastive_loss(self, seq_features, temperature=0.1):
        T = seq_features.size(0)
        if T < 2:
            return torch.tensor(0., device=seq_features.device)

        anchor = seq_features[:-1]  # [T-1, D]
        positive = seq_features[1:]  # [T-1, D]

        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)

        logits = torch.matmul(anchor, positive.t()) / temperature
        labels = torch.arange(logits.size(0), device=seq_features.device)

        return F.cross_entropy(logits, labels)

class SFF(nn.Module):
    def __init__(self, dim, freq_bands=8, reduction=16, act_layer=nn.GELU):
        super().__init__()
        self.dim = dim
        self.freq_bands = freq_bands
        self.freq_attn = nn.Sequential(
            nn.Linear(freq_bands, freq_bands),
            nn.Sigmoid()
        )
        self.freq_gate = nn.Sequential(
            nn.Linear(freq_bands, freq_bands // 2, bias=False),
            act_layer(),
            nn.Linear(freq_bands // 2, freq_bands, bias=False),
            nn.Sigmoid()
        )
        self.freq_proj = nn.Linear(freq_bands, dim)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(dim)
    def dct_features(self, x):
        x_freq = torch.from_numpy(
            dct(x.detach().cpu().numpy(), type=2, axis=-1, norm='ortho')
        ).to(x.device).type_as(x)
        freq_out = x_freq[:, :self.freq_bands]
        return freq_out
    def forward(self, x):
        freq = self.dct_features(x)
        freq_attn = self.freq_attn(freq)
        freq_weighted = freq * freq_attn
        freq_gate = self.freq_gate(freq_weighted)
        freq_emb = self.freq_proj(freq_weighted * freq_gate)
        freq_emb = self.norm(freq_emb)
        freq_emb = self.dropout(freq_emb)
        return freq_emb

# 通道注意力分支
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        w = self.fc1(x)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        return x * w

# 空域MLP分支
class SpatialMLPEnhance(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        out = self.mlp(x)
        out = self.norm(out)
        out = self.dropout(out)
        return out
    
    
class FusedFeatureEnhance(nn.Module):
    def __init__(self, dim, freq_bands=8, reduction=8):
        super().__init__()
        self.freq_enhance = SFF(dim, freq_bands, reduction)
        self.se_enhance = SEBlock(dim, reduction)
        self.spa_enhance = SpatialMLPEnhance(dim)
        # 融合MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.Dropout(0.1)
        )
    def forward(self, x):
        # 三路增强
        freq_feat = self.freq_enhance(x)
        se_feat = self.se_enhance(x)
        spa_feat = self.spa_enhance(x)
        # 拼接融合
        fusion = torch.cat([freq_feat, se_feat, spa_feat], dim=-1)
        out = self.fusion_mlp(fusion)
        # 残差
        return out + x

class OutputsForMetrics(nn.Module):
    def __init__(self):
        super().__init__()
        self.probs = CatMetric()
        self.labels = CatMetric()
        self.idx = CatMetric()

    def reset(self):
        self.probs.reset()
        self.labels.reset()
        self.idx.reset()

@dataclass
class Batch:
    images: None | torch.Tensor
    labels: None | torch.Tensor
    identity: None | torch.Tensor
    source: None | torch.Tensor
    idx: None | torch.Tensor
    paths: None | list[str]

    def __getitem__(self, key):
        # if batch["image"] is called, return batch.images
        return getattr(self, key)

    @staticmethod
    def from_dict(batch: dict):
        return Batch(
            images=batch.get("image"),
            labels=batch.get("label"),
            identity=batch.get("identity"),
            source=batch.get("source"),
            idx=batch.get("idx"),
            paths=batch.get("path"),
        )

def slerp(A: torch.Tensor, B: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
    """
    Spherical linear interpolation between two batched points A and B on a unit hypersphere.

    Parameters:
    - A: First set of points, shape (batch_size, d).
    - B: Second set of points, shape (batch_size, d).
    - t: Interpolation parameter in range [0, 1], shape (batch_size, 1) or single value.

    Returns:
    - torch.Tensor: Interpolated points, shape (batch_size, d).
    """
    # Ensure inputs are unit vectors
    A = F.normalize(A, dim=-1)
    B = F.normalize(B, dim=-1)

    # Compute dot product for each pair of points
    dot = torch.sum(A * B, dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)  # Avoid numerical issues

    # Compute the angle for each pair
    theta = torch.acos(dot)

    # Slerp formula
    sin_theta = torch.sin(theta)
    t_theta = t * theta
    coeff_a = torch.sin(theta - t_theta) / sin_theta
    coeff_b = torch.sin(t_theta) / sin_theta

    # Compute the interpolated points
    interpolated = coeff_a * A + coeff_b * B

    return interpolated

def compute_across_videos(files: list, probs: np.ndarray, labels: np.ndarray):
    """
    Calculate mean probs for each video across all frames
    """

    # Get all before the last /
    # For example: a/b/c/d -> a/b/c
    videos = [f[: -f[::-1].find("/")] for f in files]

    # Group by video: video -> [indices]
    video2idx = {v: [] for v in videos}
    for i, v in enumerate(videos):
        video2idx[v].append(i)

    # Calculate mean probs for each video across all frames
    video2probs = {v: [] for v in videos}
    video2labels = {v: [] for v in videos}
    for v, idxs in video2idx.items():
        video2probs[v] = np.mean(probs[idxs], axis=0)
        video2labels[v] = int(labels[idxs[0]])

    video_probs = np.array(list(video2probs.values()))
    video_labels = np.array(list(video2labels.values()))

    return video_probs, video_labels

class STFFNet(pl.LightningModule):
    def __init__(self, config: Config, verbose: bool = False):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.model_dump())

        if verbose:
            logger.print(config)

        seed_everything(self.config.seed, workers=True, verbose=verbose)

        self._init_feature_extractor()
        self._init_head()
        self._freeze_parameters()
        self._init_peft()
        self._init_loss()
        self._init_metrics()
        self.temporal_module = TCET(dim=1024, nhead=8, num_layers=2, lambda_contrast=0.05)
        self.asp = SGFP(
            drop_ratio=0.2,     # 可以调整
            method='shuffle',      # 'mask'/'noise'/'mixup'任选
            mode='feature+grad' # 或 'feature'
        )

        if verbose:
            self.print_trainable_parameters()

    def _init_metrics(self):
        self.train_step_outputs = OutputsForMetrics()
        self.val_step_outputs = OutputsForMetrics()
        self.test_step_outputs = OutputsForMetrics()

    def _init_feature_extractor(self):
        backbone = self.config.backbone.lower()

        if "clip" in backbone or "FaRL" in backbone:
            if Head.needs_patches(self.config.head):
                from src.encoders.clip_encoder import CLIPEncoderPatches

                self.feature_extractor = CLIPEncoderPatches(backbone)

            else:
                from src.encoders.clip_encoder import CLIPEncoder

                self.feature_extractor = CLIPEncoder(backbone)

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # self.feature_extractor.eval()
        # self.feature_extractor.to(self.device)

    def _init_peft(self):
        if self.config.peft.enabled:
            from peft import get_peft_model

            if self.config.peft.lora is not None and self.config.peft.lora.enabled:
                from peft import LoraConfig

                peft_config = LoraConfig(
                    target_modules=self.config.peft.lora.target_modules,
                    r=self.config.peft.lora.rank,
                    lora_alpha=self.config.peft.lora.alpha,
                    lora_dropout=self.config.peft.lora.dropout,
                    bias=self.config.peft.lora.bias,
                    use_rslora=self.config.peft.lora.use_rslora,
                    use_dora=self.config.peft.lora.use_dora,
                )

            elif self.config.peft.ln_tuning is not None and self.config.peft.ln_tuning.enabled:
                from peft import LNTuningConfig

                peft_config = LNTuningConfig(target_modules=self.config.peft.ln_tuning.target_modules)

            else:
                raise ValueError("Unknown PEFT configuration")

            backbone = self.feature_extractor
            training_parameters = {name for name, param in backbone.named_parameters() if param.requires_grad}

            self.feature_extractor = get_peft_model(self.feature_extractor, peft_config)

            for name, param in backbone.named_parameters():
                if name in training_parameters:
                    param.requires_grad = True

    def _init_head(self):
        features_dim = self.feature_extractor.get_features_dim()
        self.feature_enhance = FusedFeatureEnhance(dim=features_dim, freq_bands=8, reduction=8)
        self.model = head.LinearProbe(features_dim, self.config.num_classes)

        match self.config.head:
            case Head.Linear:
                self.model = head.LinearProbe(features_dim, self.config.num_classes)

            case Head.LinearNorm:
                self.model = head.LinearProbe(features_dim, self.config.num_classes, True)

            case _:
                raise ValueError(f"Unknown head: {self.config.head}")

        # self.model.eval()
        # self.model.to(self.device)

    def _freeze_parameters(self):
        # Freeze feature extractor
        self.feature_extractor.requires_grad_(not self.config.freeze_feature_extractor)

        if len(self.config.unfreeze_layers) > 0:
            for name, param in self.named_parameters():
                if any(layer in name for layer in self.config.unfreeze_layers):
                    param.requires_grad = True

    def print_trainable_parameters(self):
        logger.print("\n🔥 [red bold]Trainable parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                logger.print(f"[red]{name} shape = {tuple(param.shape)}")

        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.print(
            f"Total parameters: {all_params}, trainable: {trainable_params}, %: {trainable_params / all_params * 100:.4f}"
        )

    def _init_loss(self):
        self.criterion = Loss(self.config.loss)

    def get_preprocessing(self) -> Callable[[Image.Image], torch.Tensor]:
        return self.feature_extractor.preprocess

    def forward(self, inputs) -> head.HeadOutput:
        features = self.feature_extractor(inputs)
        if features.dim() == 3:
            features = features.mean(dim=1)
        features = self.feature_enhance(features)
        outputs = self.model(features)

        return outputs

    def log_loss(self, loss: LossOutputs, stage: str):
        if loss.total is not None:
            self.log(f"{stage}/loss", loss.total, prog_bar=True, on_epoch=True)
        if loss.ce_labels is not None:
            self.log(f"{stage}/loss_ce", loss.ce_labels, prog_bar=True, on_epoch=True)

    def log_aliunif(self, outputs: head.HeadOutput, labels: torch.Tensor, stage: str):
        alignment = unifalign.alignment(outputs.features, labels)
        uniformity = unifalign.uniformity(outputs.features)
        self.log(f"{stage}/alignment", alignment, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/uniformity", uniformity, prog_bar=True, on_epoch=True)

    def get_probs(self, outputs: head.HeadOutput):
        return outputs.logits_labels.softmax(1)

    def get_batch(self, batch: dict) -> Batch:
        return Batch.from_dict(batch)

    def slerp_feature_augmentation(self, batch: Batch, features: torch.Tensor):
        # Perform slerp on features, each class independently, vectorized

        if self.training and self.config.slerp_feature_augmentation:
            labels = batch.labels

            # Iterate over each unique class label
            for class_label in torch.unique(labels):
                class_mask = labels == class_label

                # If there are fewer than 2 features for the class, skip slerp
                if class_mask.sum() < 2:
                    continue

                # Get the features for the current class
                class_features = features[class_mask]

                # Sample pairs of embeddings from the current class
                num_embeddings = len(class_features)
                indices2 = torch.randperm(num_embeddings)
                A = class_features
                B = class_features[indices2]

                # Generate a random interpolation parameter t for each embedding in the batch
                t = torch.rand((num_embeddings, 1), device=features.device, dtype=features.dtype)

                # Extend range from [0, 1] to [t0, t1]
                t0, t1 = self.config.slerp_feature_augmentation_range
                t = t * (t1 - t0) + t0

                # autocast
                augmented_embeddings = slerp(A, B, t)  # Perform slerp

                # Update the features for the current class
                features[class_mask] = augmented_embeddings.to(features.dtype)

        return features

    def training_step(self, batch, batch_idx):
        batch = self.get_batch(batch)
        features = self.feature_extractor(batch.images)
        features = self.slerp_feature_augmentation(batch, features)

        # ====== 时序Transformer增强 ======
        temporal_loss, enhanced_features = self.temporal_module(features, batch.idx)
        self.log('train/temporal_loss', temporal_loss, on_step=True, on_epoch=True)
        # 用增强特征
        outputs = self.model(enhanced_features)
        # =================================

        # ==== ASP扰动损失（只在训练阶段）====
        asp_loss = 0.0
        if self.training and hasattr(self, "asp"):
            # logits_labels要有梯度，所以不用 no_grad
            asp_perturbed, _ = self.asp(
                enhanced_features, 
                outputs.logits_labels, 
                batch.labels
            )
            asp_outputs = self.model(asp_perturbed)
            asp_loss = F.cross_entropy(asp_outputs.logits_labels, batch.labels)
        # =================================

        # 主loss
        loss_inputs = LossInputs(
            logits_labels=outputs.logits_labels,
            labels=batch.labels,
            embeddings=outputs.features,
        )
        main_loss = self.criterion(loss_inputs)
        total_loss = main_loss.total + 0.3 * temporal_loss + 0.5 * asp_loss  # ASP损失加权

        probs = self.get_probs(outputs)
        self.log_loss(main_loss, "train")
        self.log_aliunif(outputs, batch.labels, "train")
        self.train_step_outputs.labels.update(batch.labels)
        self.train_step_outputs.probs.update(probs.detach())
        self.train_step_outputs.idx.update(batch.idx)

        return total_loss

    def on_train_start(self):
        logger.print(f"[blue]Logs: {self.logger.log_dir}")
        self.log("num_train_files", len(self.trainer.datamodule.train_dataset))
        self.log("num_val_files", len(self.trainer.datamodule.val_dataset))

    def on_test_start(self):
        logger.print(f"[blue]Logs: {self.logger.log_dir}")
        self.log("num_test_files", len(self.trainer.datamodule.test_dataset))

    def sources_probs_to_binary(self, probs: np.ndarray) -> np.ndarray:
        # probs[:, 0]  # is real probs
        # probs[:, 1:]  # is fake probs (for each generator)
        return np.stack([probs[:, 0], probs[:, 1:].max(axis=1)], 1)

    def log_metrics(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        stage: Literal["train", "test", "val"],
        prefix: str,
        level: Literal["frame", "video"],
        dataset: BaseDataset,
    ):
        """
        Images are saved to
        `log_dir / prefix / level_metrics / metric.png`
        """

        log_dir = self.logger.log_dir

        Stage = stage.capitalize()

        # Compute ROC and PR curves for every class
        fprs, tprs, roc_ths, ovr_macro_auroc = metrics.ovr_roc(labels, probs)
        precs, recs, pr_ths, ovr_macro_ap = metrics.ovr_prc(labels, probs)

        # Compute EER (Equal Error Rate)
        if self.config.num_classes == 2:
            eer = metrics.calculate_eer(labels, probs)
            self.log(f"{prefix}/eer_{level}", eer)

        # Compute predictions by argmax rule
        preds = probs.argmax(1)

        # Log metrics
        self.log(f"{prefix}/auroc_{level}", ovr_macro_auroc)
        self.log(f"{prefix}/acc_{level}", M.accuracy_score(labels, preds))
        self.log(f"{prefix}/balanced_acc_{level}", M.balanced_accuracy_score(labels, preds))
        self.log(f"{prefix}/f1_score_{level}", M.f1_score(labels, preds, average="macro"))
        self.log(f"{prefix}/mAP_{level}", ovr_macro_ap)

        class_names = dataset.get_class_names()

        plots.plot_probs_distribution(
            probs,
            labels,
            class_names,
            f"{log_dir}/{prefix}/{level}_metrics/{stage}_probs_distribution.png",
        )

        plots.plot_roc_curve(
            fprs,
            tprs,
            roc_ths,
            f"{Stage} ROC ({level}-level)",
            f"{log_dir}/{prefix}/{level}_metrics/{stage}_roc_{level}.png",
            0.01,
            class_names,
        )

        plots.plot_prc_curve(
            precs,
            recs,
            pr_ths,
            f"{Stage} PR Curve ({level}-level)",
            f"{log_dir}/{prefix}/{level}_metrics/{stage}_pr_curve.png",
            0.01,
            class_names,
        )

        plots.plot_f1_curve(
            precs,
            recs,
            pr_ths,
            f"{Stage} F1 Curve ({level}-level)",
            f"{log_dir}/{prefix}/{level}_metrics/{stage}_f1_curve.png",
            0.01,
            class_names,
        )

        # Confusion matrix
        conf = M.confusion_matrix(labels, preds)
        plots.plot_confusion_matrix(
            conf,
            class_names,
            f"{Stage} Confusion Matrix ({level}-level)",
            f"{log_dir}/{prefix}/{level}_metrics/{stage}_confusion.png",
        )
        plots.plot_confusion_matrix(
            conf,
            class_names,
            f"{Stage} Confusion Matrix ({level}-level)",
            f"{log_dir}/{prefix}/{level}_metrics/{stage}_confusion_norm.png",
            True,
        )

        if any(isinstance(l, WandbLogger) for l in self.loggers):
            wandb_logger = [l for l in self.loggers if isinstance(l, WandbLogger)][0]

            wandb_logger.log_metrics(
                {
                    f"confusion/{stage}_{level}": wandb.plot.confusion_matrix(
                        probs=probs,
                        y_true=labels,
                        class_names=["real", "fake"],
                        title=f"{Stage} Confusion Matrix {level.capitalize()}",
                    )
                }
            )

    def log_all_metrics(
        self,
        outputs_for_metrics: OutputsForMetrics,
        stage: Literal["train", "test", "val"],
        dataset: BaseDataset,
    ):
        # Merge all predictions and labels across processes
        labels = outputs_for_metrics.labels.compute().cpu().int().numpy()
        probs = outputs_for_metrics.probs.compute().cpu().numpy()
        idx = outputs_for_metrics.idx.compute().cpu().int().numpy()
        files = [dataset.files[i] for i in idx]  # Get files in the same order as the rest
        outputs_for_metrics.reset()

        if self.config.make_binary_before_video_aggregation:
            if probs.shape[1] > 2:
                probs = self.sources_probs_to_binary(probs)

        # Compute probs and labels for videos
        video_probs, video_labels = compute_across_videos(files, probs, labels)

        # Convery to binary if sources are used
        if not self.config.make_binary_before_video_aggregation:
            if probs.shape[1] > 2:
                probs = self.sources_probs_to_binary(probs)
                video_probs = self.sources_probs_to_binary(video_probs)

        self.log_metrics(probs, labels, stage, stage, "frame", dataset)
        self.log_metrics(video_probs, video_labels, stage, stage, "video", dataset)

        # if trn_files / val_files / tst_files is dict, separate metrics for each dataset
        if dataset.dataset2files is not None:
            if not self.config.make_binary_before_video_aggregation:
                logger.print_warning(
                    "`make_binary_before_video_aggregation=False` is not supported when trn_files / val_files / tst_files is dict"
                )

            file2index = {f: i for i, f in enumerate(files)}
            for dataset_name, dataset_files in dataset.dataset2files.items():
                # Get files only for current dataset
                dataset_files = np.intersect1d(files, dataset_files)
                file_indices = [file2index[f] for f in dataset_files]
                dataset_probs = probs[file_indices]
                dataset_labels = labels[file_indices]
                dataset_files = [files[i] for i in file_indices]

                self.log_metrics(
                    dataset_probs,
                    dataset_labels,
                    stage,
                    f"{stage}/dataset/{dataset_name}",
                    "frame",
                    dataset,
                )

                dataset_video_probs, dataset_video_labels = compute_across_videos(
                    dataset_files, dataset_probs, dataset_labels
                )

                self.log_metrics(
                    dataset_video_probs,
                    dataset_video_labels,
                    stage,
                    f"{stage}/dataset/{dataset_name}",
                    "video",
                    dataset,
                )

    def on_train_epoch_end(self):
        if self.logger.log_dir is None:
            # TODO: figure out why logger.log_dir can be None
            return

        # Log learning rate
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])

        # Log weights norms
        try:
            self.log("model/linear-W-norm", self.model.linear.weight.norm().item())
            self.log("model/linear-b-norm", self.model.linear.bias.norm().item())
        except Exception:
            pass

        dataset = self.trainer.datamodule.train_dataset
        self.log_all_metrics(self.train_step_outputs, "train", dataset)

    def validation_step(self, batch, batch_idx):
        batch = self.get_batch(batch)
        outputs = self.forward(batch.images)
        loss_inputs = LossInputs(
            logits_labels=outputs.logits_labels,
            labels=batch.labels,
            embeddings=outputs.features,
        )
        loss = self.criterion(loss_inputs)
        probs = self.get_probs(outputs)

        self.log_loss(loss, "val")
        self.log_aliunif(outputs, batch.labels, "val")
        self.val_step_outputs.labels.update(batch.labels)
        self.val_step_outputs.probs.update(probs.detach())
        self.val_step_outputs.idx.update(batch.idx)

    def on_validation_epoch_end(self):
        if self.logger.log_dir is None:
            # TODO: figure out why logger.log_dir can be None
            return

        dataset = self.trainer.datamodule.val_dataset
        self.log_all_metrics(self.val_step_outputs, "val", dataset)

    def test_step(self, batch, batch_idx):
        batch = self.get_batch(batch)
        outputs = self.forward(batch.images)
        loss_inputs = LossInputs(
            logits_labels=outputs.logits_labels,
            labels=batch.labels,
            embeddings=outputs.features,
        )
        loss = self.criterion(loss_inputs)
        probs = self.get_probs(outputs)

        self.log_loss(loss, "test")
        self.log_aliunif(outputs, batch.labels, "test")
        self.test_step_outputs.labels.update(batch.labels)
        self.test_step_outputs.probs.update(probs.detach())
        self.test_step_outputs.idx.update(batch.idx)

    def on_test_epoch_end(self):
        if self.logger.log_dir is None:
            # TODO: figure out why logger.log_dir can be None
            return

        # Concatenate all predictions and labels
        probs = self.test_step_outputs.probs.compute().cpu().numpy()
        labels = self.test_step_outputs.labels.compute().cpu().int().numpy()
        idx = self.test_step_outputs.idx.compute().cpu().int().numpy()

        dataset = self.trainer.datamodule.test_dataset

        files = [dataset.files[i] for i in idx]

        # preds is a 2D array of shape (num_samples, num_classes)
        probs = {f"prob_class_{i}": np.round(probs[:, i], 4) for i in range(probs.shape[1])}
        table = pd.DataFrame({"files": files, "labels": labels, **probs})

        # Save to CSV
        table.to_csv(f"{self.logger.log_dir}/test_predictions.csv", index=False, float_format="%.4f")

        self.log_all_metrics(self.test_step_outputs, "test", dataset)

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()  # because we need an access to the dataloader

        # Separate parameters for weight decay and no weight decay
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

            optimizer_grouped_parameters = [
                {"params": decay_params, "weight_decay": self.config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]

        # Configure optimizer
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
        )

        optimizers = {"optimizer": optimizer}

        # Configure LR scheduler
        if self.config.lr_scheduler == "cosine":
            #! be careful when running experiments with limit_train_batches
            if self.config.limit_train_batches is not None:
                logger.print_warning_once("lr scheduling and limit_train_batches are not compatible")
            T_max = self.config.max_epochs * len(self.trainer.train_dataloader)
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=self.config.min_lr)

            optimizers["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }

        return optimizers







