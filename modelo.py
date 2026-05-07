"""
model.py — MiniSentimentBERT
Arquitectura transformer encoder-only para sentiment analysis (IMDB).

Inspirada en DistilBERT, con las siguientes decisiones de diseño propias:
  - Pre-Layer Normalization (más estable que post-norm de BERT/DistilBERT)
  - Clasificador de dos capas con GELU
  - Hiperparámetros reducidos por defecto (4 capas, 256 dim) para benchmarking rápido
  - Sin token type embeddings (innecesarios para clasificación de secuencia única)
  - Scheduler coseno con warmup lineal
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import Accuracy


# ─────────────────────────────────────────────
#  MULTI-HEAD SELF-ATTENTION
# ─────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    Atención multi-cabeza estándar con máscara de padding.

    Proyecciones Q, K, V independientes → scores escalados → softmax → output.
    La escala 1/√d_k evita gradientes pequeños con dimensiones altas.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) debe ser divisible por num_heads ({num_heads})"

        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        # Proyectar y reorganizar en cabezas: (B, H, T, D)
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Scores de atención escalados: (B, H, T, T)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Enmascarar posiciones de padding con -inf para que softmax → 0
        if attention_mask is not None:
            # attention_mask: (B, T) con 1=token real, 0=padding
            mask = attention_mask[:, None, None, :]          # (B, 1, 1, T)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Combinar cabezas y proyectar: (B, T, C)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))


# ─────────────────────────────────────────────
#  FEED-FORWARD NETWORK
# ─────────────────────────────────────────────

class FeedForwardNetwork(nn.Module):
    """
    FFN de dos capas con activación GELU.

    Igual que BERT/DistilBERT: expande a ffn_dim y vuelve a hidden_dim.
    GELU es más suave que ReLU y empíricamente mejor en NLP.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
#  BLOQUE TRANSFORMER (PRE-NORM)
# ─────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Bloque encoder con Pre-Layer Normalization.

    Diferencia clave respecto a BERT/DistilBERT (post-norm):
        Post-norm:  x = LayerNorm(x + Sublayer(x))   ← BERT
        Pre-norm:   x = x + Sublayer(LayerNorm(x))   ← AQUÍ

    Pre-norm produce gradientes más estables durante el entrenamiento,
    especialmente en modelos profundos o con lr alto.
    """

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn  = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn   = FeedForwardNetwork(hidden_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask)   # Atención con residual
        x = x + self.ffn(self.norm2(x))                    # FFN con residual
        return x


# ─────────────────────────────────────────────
#  MODELO COMPLETO
# ─────────────────────────────────────────────

class MiniSentimentBERT(nn.Module):
    """
    Transformer encoder-only para clasificación binaria de sentimiento.

    Arquitectura:
        Embeddings (token + posición) → N × TransformerBlock → [CLS] → Clasificador

    El token [CLS] es el índice 2 del vocabulario (según el tokenizer BertWordPiece
    entrenado en data.py con special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]).
    Al igual que en BERT, extraemos su representación final para la clasificación.

    Parámetros por defecto (modo ligero para benchmarking):
        vocab_size  = 10_000   (tamaño del vocabulario personalizado)
        hidden_dim  = 256      (DistilBERT usa 768)
        num_heads   = 8        (head_dim = 32)
        num_layers  = 4        (DistilBERT usa 6)
        ffn_dim     = 1024     (DistilBERT usa 3072)
        max_len     = 256      (longitud máxima de secuencia)
        dropout     = 0.1
        num_classes = 2        (positivo / negativo)
    """

    def __init__(
        self,
        vocab_size:  int   = 10_000,
        hidden_dim:  int   = 256,
        num_heads:   int   = 8,
        num_layers:  int   = 4,
        ffn_dim:     int   = 1024,
        max_len:     int   = 256,
        dropout:     float = 0.1,
        num_classes: int   = 2,
    ):
        super().__init__()

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)  # 0 = [PAD]
        self.pos_emb   = nn.Embedding(max_len, hidden_dim)
        self.emb_norm  = nn.LayerNorm(hidden_dim)
        self.emb_drop  = nn.Dropout(dropout)

        # Bloques transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Normalización final (buena práctica con pre-norm)
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Clasificador de dos capas sobre el token [CLS]
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Inicialización de pesos estilo BERT
        self._init_weights()

    def _init_weights(self):
        """Inicialización normal truncada (estándar en transformers)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device

        # Embeddings: token + posición
        pos = torch.arange(T, device=device).unsqueeze(0)          # (1, T)
        x = self.token_emb(input_ids) + self.pos_emb(pos)          # (B, T, C)
        x = self.emb_drop(self.emb_norm(x))

        # Pasar por los bloques transformer
        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.final_norm(x)

        # Extraer representación del token [CLS] (posición 0)
        cls_repr = x[:, 0, :]                                       # (B, C)
        return self.classifier(cls_repr)                            # (B, num_classes)


# ─────────────────────────────────────────────
#  MÓDULO LIGHTNING
# ─────────────────────────────────────────────

class IMDBModel(L.LightningModule):
    """
    Wrapper Lightning para MiniSentimentBERT.

    Incluye:
        - Logging de loss y accuracy en train/val/test
        - Optimizador AdamW con weight decay
        - Scheduler coseno con warmup lineal
    """

    def __init__(
        self,
        vocab_size:  int   = 10_000,
        hidden_dim:  int   = 256,
        num_heads:   int   = 8,
        num_layers:  int   = 4,
        ffn_dim:     int   = 1024,
        max_len:     int   = 256,
        dropout:     float = 0.1,
        lr:          float = 2e-4,
        warmup_steps: int  = 200,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MiniSentimentBERT(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            max_len=max_len,
            dropout=dropout,
        )

        self.loss_fn  = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="binary")
        self.val_acc   = Accuracy(task="binary")
        self.test_acc  = Accuracy(task="binary")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, attention_mask)

    # ── Paso compartido ──────────────────────
    def _shared_step(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss   = self.loss_fn(logits, batch["label"])
        preds  = logits.argmax(dim=-1)
        return loss, preds, batch["label"]

    # ── Pasos de train / val / test ──────────
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, preds, labels = self._shared_step(batch)
        self.train_acc(preds, labels)
        self.log("train/loss", loss,           on_step=True,  on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, preds, labels = self._shared_step(batch)
        self.val_acc(preds, labels)
        self.log("val/loss", loss,          prog_bar=True)
        self.log("val/acc",  self.val_acc,  prog_bar=True)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        loss, preds, labels = self._shared_step(batch)
        self.test_acc(preds, labels)
        self.log("test/loss", loss)
        self.log("test/acc",  self.test_acc)

    # ── Optimizador + Scheduler ──────────────
    def configure_optimizers(self):
        """
        AdamW con weight decay selectivo: no se aplica a bias ni LayerNorm.
        Scheduler: warmup lineal → decaimiento coseno.
        """
        decay_params     = []
        no_decay_params  = []

        for name, param in self.named_parameters():
            if any(nd in name for nd in ["bias", "norm"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params,    "weight_decay": 0.01},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        def lr_lambda(current_step: int) -> float:
            warmup = self.hparams.warmup_steps
            if current_step < warmup:
                return float(current_step) / float(max(1, warmup))
            # Decaimiento coseno tras el warmup
            progress = (current_step - warmup) / max(1, self.trainer.estimated_stepping_batches - warmup)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "step",   # actualizar cada step, no cada época
            },
        }