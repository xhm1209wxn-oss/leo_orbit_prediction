"""
用于轨道误差预测的神经网络模型
包含 Attention-LSTM、Transformer Encoder 与 TCN 架构
"""

import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Mapping, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AttentionLSTM(nn.Module):
    """
    面向 TLE 误差预测的 Attention-LSTM 模型

    结构：
    - 双向 LSTM 层
    - 多头缩放点积注意力池化（Multi-Head Attention Pooling）
    - 用于回归的全连接层

    输入：19 维特征（ECI坐标系）
    输出：3 维 RTN 误差 [R, T, N]
    """

    def __init__(self,
                 input_size: int = 19,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 output_size: int = 3,
                 dropout: float = 0.2,
                 bidirectional: bool = True,
                 use_attention: bool = True,
                 num_heads: int = 4,
                 attn_dropout: float = 0.1,
                 max_seq_len: int = 64):
        super(AttentionLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        lstm_input_size = input_size

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        if self.use_attention:
            if lstm_output_size % num_heads != 0:
                raise ValueError(
                    f"lstm_output_size ({lstm_output_size}) 必须能被 num_heads ({num_heads}) 整除"
                )
            self.mha = nn.MultiheadAttention(
                embed_dim=lstm_output_size,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True
            )
        else:
            self.mha = None

        self.fc1 = nn.Linear(lstm_output_size, lstm_output_size // 2)
        self.fc2 = nn.Linear(lstm_output_size // 2, lstm_output_size // 4)
        self.fc3 = nn.Linear(lstm_output_size // 4, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        logger.info(
            f"Initialized Attention-LSTM: input={input_size}, hidden={hidden_size}, "
            f"layers={num_layers}, output={output_size}, bidirectional={bidirectional}, "
            f"use_attention={use_attention}, num_heads={num_heads}, attn_dropout={attn_dropout}"
        )

    @staticmethod
    def _build_padding_mask(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        mask = x.abs().sum(dim=-1).eq(0)
        all_pad_rows = mask.all(dim=1)
        if all_pad_rows.any():
            mask = mask.clone()
            mask[all_pad_rows, -1] = False
        return mask

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        lstm_out, _ = self.lstm(x)
        return lstm_out

    def _build_endpoint_query(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        从样本自身序列端点构造动态 query。
        与无注意力分支保持一致：双向时拼接前向末端 + 后向首端。
        """
        if self.bidirectional:
            return torch.cat([
                lstm_out[:, -1, :self.hidden_size],
                lstm_out[:, 0, self.hidden_size:]
            ], dim=-1)
        return lstm_out[:, -1, :]

    @staticmethod
    def _masked_max_pool(lstm_out: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        对序列输出做掩码最大池化，忽略 padding 时间步。
        """
        fill_value = torch.finfo(lstm_out.dtype).min
        masked = lstm_out.masked_fill(key_padding_mask.unsqueeze(-1), fill_value)
        return masked.max(dim=1).values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        key_padding_mask = self._build_padding_mask(x)
        lstm_out = self._encode(x)

        if self.use_attention:
            query = self._build_endpoint_query(lstm_out).unsqueeze(1)
            context, _ = self.mha(
                query, lstm_out, lstm_out,
                key_padding_mask=key_padding_mask
            )
            context = context.squeeze(1)
        else:
            context = self._masked_max_pool(lstm_out, key_padding_mask)

        out = self.relu(self.fc1(context))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)

        return out

class TransformerRegressor(nn.Module):
    """
    基于 Transformer Encoder 的序列回归模型

    输入：形状必须为 (batch, seq_len, input_size)
    输出：形状为 (batch, output_size)
    """

    def __init__(self,
                 input_size: int = 19,
                 d_model: int = 128,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 ffn_dim: int = 256,
                 output_size: int = 3,
                 attn_dropout: float = 0.1,
                 ffn_dropout: float = 0.1,
                 dropout: float = 0.1,
                 pooling: str = "mean",
                 max_seq_len: int = 64):
        super().__init__()

        if d_model < 1:
            raise ValueError("d_model must be at least 1")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if num_heads < 1:
            raise ValueError("num_heads must be at least 1")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if ffn_dim < 1:
            raise ValueError("ffn_dim must be at least 1")
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be one of {'mean', 'cls'}")
        if max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2 for Transformer")

        self.input_size = input_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.output_size = output_size
        self.pooling = pooling
        self.max_seq_len = max_seq_len

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        nn.init.trunc_normal_(self.pos_embedding.weight, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True
        )
        encoder_layer.dropout = nn.Dropout(ffn_dropout)
        encoder_layer.dropout2 = nn.Dropout(ffn_dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if self.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        hidden1 = max(1, d_model // 2)
        hidden2 = max(1, d_model // 4)
        self.fc1 = nn.Linear(d_model, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        logger.info(
            f"Initialized TransformerRegressor: input={input_size}, d_model={d_model}, "
            f"layers={num_layers}, heads={num_heads}, ffn_dim={ffn_dim}, output={output_size}, "
            f"pooling={pooling}, attn_dropout={attn_dropout}, ffn_dropout={ffn_dropout}"
        )

    @staticmethod
    def _build_padding_mask(x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"TransformerRegressor expects 3D input (batch, seq_len, features), got shape={tuple(x.shape)}"
            )
        mask = x.abs().sum(dim=-1).eq(0)
        all_pad_rows = mask.all(dim=1)
        if all_pad_rows.any():
            mask = mask.clone()
            mask[all_pad_rows, -1] = False
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"TransformerRegressor requires real sequence input (batch, seq_len, features), got shape={tuple(x.shape)}"
            )
        if x.size(-1) != self.input_size:
            raise ValueError(
                f"TransformerRegressor expected input_size={self.input_size}, got last dim={x.size(-1)}"
            )

        key_padding_mask = self._build_padding_mask(x)

        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len})"
            )

        x = self.input_proj(x)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(pos_ids)

        if self.pooling == "cls":
            bsz = x.size(0)
            cls = self.cls_token.expand(bsz, -1, -1)
            x = torch.cat([cls, x], dim=1)
            cls_mask = torch.zeros((bsz, 1), dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        if self.pooling == "cls":
            context = x[:, 0, :]
        else:
            valid = (~key_padding_mask).unsqueeze(-1).to(x.dtype)
            denom = valid.sum(dim=1).clamp(min=1.0)
            context = (x * valid).sum(dim=1) / denom

        out = self.relu(self.fc1(context))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out


class _TemporalBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int,
                 dropout: float,
                 use_residual: bool,
                 causal: bool):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout_p = dropout
        self.use_residual = use_residual
        self.causal = causal

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if use_residual and in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = None

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        total_pad = self.dilation * (self.kernel_size - 1)
        if self.causal:
            return F.pad(x, (total_pad, 0))

        left = total_pad // 2
        right = total_pad - left
        return F.pad(x, (left, right))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self._pad(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self._pad(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)

        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            out = out + residual

        return self.relu(out)


class TCNRegressor(nn.Module):
    """
    基于 TCN 的序列回归模型。

    输入：形状必须为 (batch, seq_len, input_size)
    输出：形状为 (batch, output_size)
    """

    def __init__(self,
                 input_size: int = 19,
                 channels: List[int] = None,
                 kernel_size: int = 3,
                 dilation_base: int = 2,
                 output_size: int = 3,
                 dropout: float = 0.1,
                 use_residual: bool = True,
                 causal: bool = True,
                 pooling: str = "last"):
        super().__init__()

        channels = channels or [128, 128, 128]

        if not channels:
            raise ValueError("channels must not be empty")
        if any(c < 1 for c in channels):
            raise ValueError("all channels must be >= 1")
        if kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        if dilation_base < 1:
            raise ValueError("dilation_base must be >= 1")
        if pooling not in {"last", "mean"}:
            raise ValueError("pooling must be one of {'last', 'mean'}")

        self.input_size = input_size
        self.channels = [int(c) for c in channels]
        self.kernel_size = int(kernel_size)
        self.dilation_base = int(dilation_base)
        self.output_size = output_size
        self.use_residual = bool(use_residual)
        self.causal = bool(causal)
        self.pooling = pooling

        self.input_proj = nn.Conv1d(input_size, self.channels[0], kernel_size=1)

        blocks = []
        in_ch = self.channels[0]
        for idx, out_ch in enumerate(self.channels):
            dilation = self.dilation_base ** idx
            blocks.append(_TemporalBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=self.kernel_size,
                dilation=dilation,
                dropout=dropout,
                use_residual=self.use_residual,
                causal=self.causal,
            ))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        hidden1 = max(1, self.channels[-1] // 2)
        hidden2 = max(1, self.channels[-1] // 4)
        self.fc1 = nn.Linear(self.channels[-1], hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        logger.info(
            "Initialized TCNRegressor: input=%d, channels=%s, kernel=%d, dilation_base=%d, "
            "output=%d, pooling=%s, residual=%s, causal=%s",
            input_size,
            self.channels,
            self.kernel_size,
            self.dilation_base,
            output_size,
            self.pooling,
            self.use_residual,
            self.causal,
        )

    @staticmethod
    def _build_padding_mask(x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"TCNRegressor expects 3D input (batch, seq_len, features), got shape={tuple(x.shape)}"
            )
        mask = x.abs().sum(dim=-1).eq(0)
        all_pad_rows = mask.all(dim=1)
        if all_pad_rows.any():
            mask = mask.clone()
            mask[all_pad_rows, -1] = False
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"TCNRegressor requires sequence input (batch, seq_len, features), got shape={tuple(x.shape)}"
            )
        if x.size(-1) != self.input_size:
            raise ValueError(
                f"TCNRegressor expected input_size={self.input_size}, got last dim={x.size(-1)}"
            )

        key_padding_mask = self._build_padding_mask(x)

        x = x.transpose(1, 2)            # (batch, features, seq_len)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)            # (batch, seq_len, channels)

        if self.pooling == "last":
            valid_lengths = (~key_padding_mask).sum(dim=1).clamp(min=1) - 1
            gather_index = valid_lengths.view(-1, 1, 1).expand(-1, 1, x.size(-1))
            context = x.gather(dim=1, index=gather_index).squeeze(1)
        else:
            valid = (~key_padding_mask).unsqueeze(-1).to(x.dtype)
            denom = valid.sum(dim=1).clamp(min=1.0)
            context = (x * valid).sum(dim=1) / denom

        out = self.relu(self.fc1(context))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out


def _as_plain_dict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "__dict__"):
        return {
            key: value
            for key, value in vars(obj).items()
            if not key.startswith("_")
        }
    raise ValueError(f"Cannot convert object to dict: type={type(obj)}")


def _read_value(obj: Any, key: str, section: str) -> Any:
    if isinstance(obj, Mapping):
        if key not in obj:
            raise ValueError(f"Missing key '{key}' in {section}")
        return obj[key]
    if hasattr(obj, key):
        return getattr(obj, key)
    raise ValueError(f"Missing attribute '{key}' in {section}")


def _resolve_model_type(config) -> str:
    model_type = str(getattr(config.model, 'type', '')).lower()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model.type: {model_type}. Expected one of {sorted(MODEL_REGISTRY.keys())}."
        )
    return model_type


def _get_model_common(config):
    if not hasattr(config.model, 'common'):
        raise ValueError("Missing model.common in config. This version only supports decoupled schema.")
    return config.model.common


def _get_model_params(config):
    if not hasattr(config.model, 'params'):
        raise ValueError("Missing model.params in config. This version only supports decoupled schema.")
    return config.model.params


def _build_lstm_kwargs(common: Any, params: Any) -> Dict[str, Any]:
    return {
        'input_size': int(_read_value(common, 'input_size', 'model.common')),
        'hidden_size': int(_read_value(params, 'hidden_size', 'model.params')),
        'num_layers': int(_read_value(params, 'num_layers', 'model.params')),
        'output_size': int(_read_value(common, 'output_size', 'model.common')),
        'dropout': float(_read_value(common, 'dropout', 'model.common')),
        'bidirectional': bool(_read_value(params, 'bidirectional', 'model.params')),
        'use_attention': bool(_read_value(params, 'use_attention', 'model.params')),
        'num_heads': int(_read_value(params, 'num_heads', 'model.params')),
        'attn_dropout': float(_read_value(params, 'attn_dropout', 'model.params')),
        'max_seq_len': int(_read_value(common, 'sequence_length', 'model.common')),
    }


def _build_transformer_kwargs(common: Any, params: Any) -> Dict[str, Any]:
    return {
        'input_size': int(_read_value(common, 'input_size', 'model.common')),
        'd_model': int(_read_value(params, 'd_model', 'model.params')),
        'num_layers': int(_read_value(params, 'num_layers', 'model.params')),
        'num_heads': int(_read_value(params, 'num_heads', 'model.params')),
        'ffn_dim': int(_read_value(params, 'ffn_dim', 'model.params')),
        'output_size': int(_read_value(common, 'output_size', 'model.common')),
        'attn_dropout': float(_read_value(params, 'attn_dropout', 'model.params')),
        'ffn_dropout': float(_read_value(params, 'ffn_dropout', 'model.params')),
        'dropout': float(_read_value(common, 'dropout', 'model.common')),
        'pooling': str(_read_value(params, 'pooling', 'model.params')),
        'max_seq_len': int(_read_value(common, 'sequence_length', 'model.common')),
    }


def _build_tcn_kwargs(common: Any, params: Any) -> Dict[str, Any]:
    channels = list(_read_value(params, 'channels', 'model.params'))
    return {
        'input_size': int(_read_value(common, 'input_size', 'model.common')),
        'channels': [int(c) for c in channels],
        'kernel_size': int(_read_value(params, 'kernel_size', 'model.params')),
        'dilation_base': int(_read_value(params, 'dilation_base', 'model.params')),
        'output_size': int(_read_value(common, 'output_size', 'model.common')),
        'dropout': float(_read_value(common, 'dropout', 'model.common')),
        'use_residual': bool(_read_value(params, 'use_residual', 'model.params')),
        'causal': bool(_read_value(params, 'causal', 'model.params')),
        'pooling': str(_read_value(params, 'pooling', 'model.params')),
    }


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    'lstm': {
        'model_class': AttentionLSTM,
        'builder': _build_lstm_kwargs,
    },
    'transformer': {
        'model_class': TransformerRegressor,
        'builder': _build_transformer_kwargs,
    },
    'tcn': {
        'model_class': TCNRegressor,
        'builder': _build_tcn_kwargs,
    },
}


def build_model_spec(config) -> Tuple[type, Dict[str, Any], str]:
    """
    根据解耦配置解析模型类、初始化参数与模型类型。
    """
    model_type = _resolve_model_type(config)
    common = _get_model_common(config)
    params = _get_model_params(config)

    entry = MODEL_REGISTRY[model_type]
    model_class = entry['model_class']
    builder: Callable[[Any, Any], Dict[str, Any]] = entry['builder']
    model_kwargs = builder(common, params)

    return model_class, model_kwargs, model_type


def get_model_class_and_kwargs(config) -> Tuple[type, Dict[str, Any]]:
    """
    兼容内部调用：返回模型类和参数。
    """
    model_class, model_kwargs, _ = build_model_spec(config)
    return model_class, model_kwargs


def get_model_metadata(config) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    提取模型元信息，用于 checkpoint 元数据保存。
    """
    model_type = _resolve_model_type(config)
    common = _as_plain_dict(_get_model_common(config))
    params = _as_plain_dict(_get_model_params(config))
    return model_type, common, params


def create_model(config, device: torch.device) -> nn.Module:
    """
    根据配置创建模型（LSTM / Transformer / TCN）
    """
    model_class, model_kwargs, model_type = build_model_spec(config)
    model = model_class(**model_kwargs).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model created: type=%s, class=%s, trainable parameters=%s",
        model_type,
        model_class.__name__,
        f"{num_params:,}",
    )

    return model


def _infer_checkpoint_model_type(checkpoint: Dict[str, Any]) -> str:
    ckpt_model_type = checkpoint.get('model_type')
    if ckpt_model_type is None:
        raise ValueError("Checkpoint missing required field: model_type")
    ckpt_model_type = str(ckpt_model_type).lower()
    if ckpt_model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Checkpoint model_type is unsupported: {ckpt_model_type}. "
            f"Expected one of {sorted(MODEL_REGISTRY.keys())}."
        )
    return ckpt_model_type


def load_model(model_path: str,
               config,
               device: torch.device) -> nn.Module:
    """
    从检查点加载已训练的模型
    """
    model = create_model(config, device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    if isinstance(checkpoint, dict):
        config_model_type = _resolve_model_type(config)
        ckpt_model_type = _infer_checkpoint_model_type(checkpoint)
        if ckpt_model_type != config_model_type:
            raise ValueError(
                f"Checkpoint model_type ({ckpt_model_type}) does not match config.model.type ({config_model_type})."
            )

    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Model loaded from {model_path}")

    return model
