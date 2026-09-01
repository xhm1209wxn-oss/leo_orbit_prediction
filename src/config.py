"""
配置管理模块
负责实验配置的加载与校验
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Type
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    device: str
    seed: int = 44


@dataclass
class SatelliteConfig:
    """卫星配置"""
    list_file: str  # 卫星列表文件路径


@dataclass
class SpaceTrackConfig:
    """Space-Track 认证信息"""
    username: str
    password: str


@dataclass
class TLEConfig:
    """TLE 下载配置"""
    start_date: str
    end_date: str
    max_dt_days: int
    orbit_samples_n: int = 8  # 每个 (i,j) 配对在一个轨道周期窗内均匀采样的分段数（k=0..n，共 n+1 点）
    delta_n_threshold: float = 0.003
    iqr_k: float = 3.0  # IQR异常值过滤倍数（用于样本和卫星级别过滤），设为0禁用
    iqr_metric: str = "T_median"  # 跨卫星过滤指标：T_median/T_mean/total_rms
    skip_cached: bool = False
    batch_size: int = 200
    request_delay_seconds: float = 2.0


@dataclass
class DataSplitConfig:
    """数据划分配置"""
    val_days: int = 20  # 验证集使用后N天数据
    num_workers: int = 0


@dataclass
class DataFilterConfig:
    """小样本过滤配置"""
    enabled: bool = False
    min_sat_samples_before_split: int = 0
    min_train_samples: int = 0
    min_val_samples: int = 0
    min_test_samples_for_eval: int = 0


@dataclass
class DataConfig:
    """数据配置"""
    spacetrack: SpaceTrackConfig
    tle: TLEConfig
    split: DataSplitConfig
    filter: DataFilterConfig


@dataclass
class ClusteringConfig:
    """聚类配置"""
    n_clusters: int
    feature_mode: str = "t_error_stats"
    method: str = "kmeans"
    balance_by_samples: bool = True
    balance_tolerance: float = 0.15
    balance_relax_step: float = 0.05
    balance_max_relax_steps: int = 3


@dataclass
class ModelCommonConfig:
    """模型公共参数（所有模型共享）"""
    input_size: int
    output_size: int
    sequence_length: int
    dropout: float


@dataclass
class LSTMModelParams:
    """LSTM 专属参数"""
    hidden_size: int
    num_layers: int
    bidirectional: bool
    use_attention: bool
    num_heads: int
    attn_dropout: float


@dataclass
class TransformerModelParams:
    """Transformer 专属参数"""
    d_model: int
    num_layers: int
    num_heads: int
    ffn_dim: int
    attn_dropout: float
    ffn_dropout: float
    pooling: str


@dataclass
class TCNModelParams:
    """TCN 专属参数"""
    channels: List[int]
    kernel_size: int
    dilation_base: int
    use_residual: bool
    causal: bool
    pooling: str


@dataclass
class ModelConfig:
    """模型配置（强制解耦结构）"""
    type: str                  # lstm / transformer / tcn
    common: ModelCommonConfig
    params: Any                # LSTMModelParams / TransformerModelParams / TCNModelParams
    params_by_type: Dict[str, Dict[str, Any]]


@dataclass
class CollaborativeConfig:
    """分组协同训练配置"""
    num_rounds: int
    local_epochs: int


@dataclass
class OptimizerConfig:
    """优化器配置"""
    name: str
    lr: float
    weight_decay: float


@dataclass
class LRScheduleConfig:
    """学习率调度配置"""
    type: str
    min_lr: float = 1e-5              # 最小学习率下界
    warmup_rounds: int = 0            # warmup 轮数（0 表示关闭）
    warmup_start_factor: float = 0.1  # warmup 初始学习率比例（相对 base lr）


@dataclass
class EarlyStoppingConfig:
    """提前停止配置"""
    enabled: bool
    patience: int
    min_delta: float


@dataclass
class ModelSelectionConfig:
    """选模指标配置"""
    metric: str = "val_t_rms_km"  # 可选: val_loss / val_t_rms_km
    mode: str = "min"             # 可选: min / max
    min_delta: float = 0.0


@dataclass
class ValidationConfig:
    """验证频率配置"""
    eval_every_rounds: int


@dataclass
class TrainingConfig:
    """训练配置"""
    collaborative: CollaborativeConfig
    optimizer: OptimizerConfig
    lr_schedule: LRScheduleConfig
    batch_size: int
    grad_clip: float
    early_stopping: EarlyStoppingConfig
    model_selection: ModelSelectionConfig
    validation: ValidationConfig
    loss_type: str = "loss2"  # 可选: loss1 / loss2 / loss3
    loss_delta: float = 1.0   # loss2 / loss3 的 delta 参数（> 0）
    loss3_mu: float = 0.0     # loss3 的近端约束系数（>= 0）
    loss_axis_weights: List[float] = None  # R/T/N 三轴相对权重；默认 [0.1, 0.8, 0.1]


@dataclass
class OutputPathsConfig:
    """输出路径配置"""
    data_raw: str
    data_processed: str
    models: str
    plots: str
    metrics: str
    clustering: str


@dataclass
class SaveConfig:
    """保存设置配置"""
    plots: bool
    clustering_results: bool


@dataclass
class VisualizationConfig:
    """可视化设置"""
    dpi: int
    format: str
    style: str


@dataclass
class OutputsConfig:
    """输出配置"""
    paths: OutputPathsConfig
    save: SaveConfig
    visualization: VisualizationConfig


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str
    console: bool


@dataclass
class Config:
    """完整配置"""
    experiment: ExperimentConfig
    satellite: SatelliteConfig
    data: DataConfig
    clustering: ClusteringConfig
    model: ModelConfig
    training: TrainingConfig
    outputs: OutputsConfig
    logging: LoggingConfig
    targets: Dict[str, int] = None  # 从文件加载的卫星目标


def _instantiate_dataclass_strict(dc_cls: Type, raw_dict: Dict[str, Any], section: str):
    if not isinstance(raw_dict, dict):
        raise ValueError(f"{section} must be a mapping")

    dc_fields = fields(dc_cls)
    allowed = {f.name for f in dc_fields}
    unknown = sorted(set(raw_dict.keys()) - allowed)
    if unknown:
        raise ValueError(f"{section} has unknown keys: {unknown}")

    required = {
        f.name for f in dc_fields
        if f.default is MISSING and f.default_factory is MISSING
    }
    missing = sorted(required - set(raw_dict.keys()))
    if missing:
        raise ValueError(f"{section} is missing required keys: {missing}")

    return dc_cls(**raw_dict)


def _parse_model_config(raw_model_cfg: Dict[str, Any]) -> ModelConfig:
    if not isinstance(raw_model_cfg, dict):
        raise ValueError("model must be a mapping")

    allowed_model_keys = {"type", "common", "params_by_type", "params"}
    unknown = sorted(set(raw_model_cfg.keys()) - allowed_model_keys)
    if unknown:
        raise ValueError(
            "model section must only contain keys ['type', 'common', 'params_by_type', 'params']; "
            f"unknown keys: {unknown}"
        )
    required_model_keys = {"type", "common", "params_by_type"}
    missing = sorted(required_model_keys - set(raw_model_cfg.keys()))
    if missing:
        raise ValueError(f"model section is missing required keys: {missing}")

    model_type = str(raw_model_cfg["type"]).lower()
    model_common = _instantiate_dataclass_strict(
        ModelCommonConfig,
        raw_model_cfg["common"],
        "model.common"
    )

    params_map = {
        "lstm": LSTMModelParams,
        "transformer": TransformerModelParams,
        "tcn": TCNModelParams,
    }
    params_cls = params_map.get(model_type)
    if params_cls is None:
        raise ValueError(f"model.type must be one of {sorted(params_map.keys())}")

    raw_params_by_type = raw_model_cfg["params_by_type"]
    if not isinstance(raw_params_by_type, dict):
        raise ValueError("model.params_by_type must be a mapping")

    expected_types = set(params_map.keys())
    unknown_param_types = sorted(set(raw_params_by_type.keys()) - expected_types)
    if unknown_param_types:
        raise ValueError(
            f"model.params_by_type has unknown model types: {unknown_param_types}"
        )
    missing_param_types = sorted(expected_types - set(raw_params_by_type.keys()))
    if missing_param_types:
        raise ValueError(
            f"model.params_by_type is missing required model types: {missing_param_types}"
        )

    params_by_type_obj = {
        m_type: _instantiate_dataclass_strict(
            params_map[m_type],
            raw_params_by_type[m_type],
            f"model.params_by_type.{m_type}"
        )
        for m_type in sorted(expected_types)
    }

    model_params = params_by_type_obj[model_type]
    if "params" in raw_model_cfg:
        explicit_params = _instantiate_dataclass_strict(
            params_cls,
            raw_model_cfg["params"],
            f"model.params ({model_type})"
        )
        if asdict(explicit_params) != asdict(model_params):
            raise ValueError(
                "model.params must exactly match model.params_by_type[model.type] "
                f"(model.type={model_type})"
            )
        model_params = explicit_params
    params_by_type_dict = {
        m_type: asdict(params_obj)
        for m_type, params_obj in params_by_type_obj.items()
    }

    return ModelConfig(
        type=model_type,
        common=model_common,
        params=model_params,
        params_by_type=params_by_type_dict
    )


def load_satellite_list(list_file: str, config_dir: Path) -> Dict[str, int]:
    """
    从文件中加载卫星列表

    参数：
        list_file: 卫星列表文件名
        config_dir: 包含配置文件的目录

    返回：
        {satellite_name: norad_id} 的字典
    """
    list_path = config_dir / list_file

    if not list_path.exists():
        raise FileNotFoundError(f"Satellite list file not found: {list_path}")

    satellites = {}
    with open(list_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) >= 2:
                norad_id = int(parts[-1])
                sat_name = ' '.join(parts[:-1])
                satellites[sat_name] = norad_id

    logger.info(f"Loaded {len(satellites)} satellites from {list_file}")
    return satellites


def _parse_bool(value: Any, field_name: str) -> bool:
    """
    解析配置中的布尔值，避免字符串 "false" 被 bool(...) 当作 True。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean")


def load_config(config_path: str) -> Config:
    """
    从 YAML 文件加载配置

    参数：
        config_path: 配置文件路径

    返回：
        Config 对象
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    split_cfg = config_dict['data']['split']
    filter_cfg = config_dict['data'].get('filter', {})
    model_sel_cfg = config_dict['training'].get('model_selection', {})
    log_cfg = config_dict['logging']
    tle_cfg = dict(config_dict['data']['tle'])
    if 'days_back' in tle_cfg:
        raise ValueError(
            "data.tle.days_back 已废弃，请改用 data.tle.start_date 与 data.tle.end_date（YYYY-MM-DD）"
        )

    model_cfg = _parse_model_config(config_dict['model'])
    config = Config(
        experiment=ExperimentConfig(**config_dict['experiment']),
        satellite=SatelliteConfig(**config_dict['satellite']),
        data=DataConfig(
            spacetrack=SpaceTrackConfig(**config_dict['data']['spacetrack']),
            tle=TLEConfig(**tle_cfg),
            split=DataSplitConfig(
                val_days=split_cfg.get('val_days', 20),
                num_workers=split_cfg.get('num_workers', 0)
            ),
            filter=DataFilterConfig(
                enabled=filter_cfg.get('enabled', False),
                min_sat_samples_before_split=filter_cfg.get('min_sat_samples_before_split', 0),
                min_train_samples=filter_cfg.get('min_train_samples', 0),
                min_val_samples=filter_cfg.get('min_val_samples', 0),
                min_test_samples_for_eval=filter_cfg.get('min_test_samples_for_eval', 0)
            )
        ),
        clustering=ClusteringConfig(
            n_clusters=int(config_dict['clustering']['n_clusters']),
            feature_mode=config_dict['clustering'].get('feature_mode', 't_error_stats'),
            method=config_dict['clustering'].get('method', 'kmeans'),
            balance_by_samples=_parse_bool(
                config_dict['clustering'].get('balance_by_samples', True),
                'clustering.balance_by_samples'
            ),
            balance_tolerance=float(config_dict['clustering'].get('balance_tolerance', 0.15)),
            balance_relax_step=float(config_dict['clustering'].get('balance_relax_step', 0.05)),
            balance_max_relax_steps=int(config_dict['clustering'].get('balance_max_relax_steps', 3))
        ),
        model=model_cfg,
        training=TrainingConfig(
            collaborative=CollaborativeConfig(**config_dict['training']['collaborative']),
            optimizer=OptimizerConfig(**config_dict['training']['optimizer']),
            lr_schedule=LRScheduleConfig(**config_dict['training']['lr_schedule']),
            batch_size=config_dict['training']['batch_size'],
            grad_clip=config_dict['training']['grad_clip'],
            early_stopping=EarlyStoppingConfig(**config_dict['training']['early_stopping']),
            model_selection=ModelSelectionConfig(
                metric=str(model_sel_cfg.get('metric', 'val_t_rms_km')),
                mode=str(model_sel_cfg.get('mode', 'min')),
                min_delta=float(model_sel_cfg.get('min_delta', 0.0))
            ),
            validation=ValidationConfig(**config_dict['training']['validation']),
            loss_type=str(config_dict['training'].get('loss_type', 'loss2')).lower(),
            loss_delta=float(config_dict['training'].get('loss_delta', 1.0)),
            loss3_mu=float(config_dict['training'].get('loss3_mu', 0.0)),
            loss_axis_weights=[
                float(v) for v in config_dict['training'].get('loss_axis_weights', [0.1, 0.8, 0.1])
            ]
        ),
        outputs=OutputsConfig(
            paths=OutputPathsConfig(**config_dict['outputs']['paths']),
            save=SaveConfig(**config_dict['outputs']['save']),
            visualization=VisualizationConfig(**config_dict['outputs']['visualization'])
        ),
        logging=LoggingConfig(
            level=log_cfg['level'],
            console=log_cfg['console']
        )
    )

    config.targets = load_satellite_list(config.satellite.list_file, config_path.parent)

    logger.info(f"Configuration loaded from {config_path}")
    return config


def _validate_model_specific_params(model_type: str,
                                    common: ModelCommonConfig,
                                    params: Any,
                                    params_prefix: str):
    if model_type == 'lstm':
        if not isinstance(params, LSTMModelParams):
            raise ValueError(f"{params_prefix} must be LSTMModelParams when model.type=lstm")
        if params.hidden_size < 1:
            raise ValueError(f"{params_prefix}.hidden_size must be at least 1 for LSTM")
        if params.num_layers < 1:
            raise ValueError(f"{params_prefix}.num_layers must be at least 1 for LSTM")
        if params.use_attention:
            lstm_output_size = params.hidden_size * (2 if params.bidirectional else 1)
            if params.num_heads < 1:
                raise ValueError(f"{params_prefix}.num_heads must be at least 1")
            if not (0.0 <= params.attn_dropout <= 1.0):
                raise ValueError(f"{params_prefix}.attn_dropout must be in [0, 1]")
            if lstm_output_size % params.num_heads != 0:
                raise ValueError(
                    f"{params_prefix}.num_heads ({params.num_heads}) must divide "
                    f"lstm_output_size ({lstm_output_size}) evenly"
                )
        return

    if model_type == 'transformer':
        if not isinstance(params, TransformerModelParams):
            raise ValueError(f"{params_prefix} must be TransformerModelParams when model.type=transformer")
        if common.sequence_length < 2:
            raise ValueError("model.common.sequence_length must be at least 2 when model.type=transformer")
        if params.d_model < 1:
            raise ValueError(f"{params_prefix}.d_model must be at least 1")
        if params.num_layers < 1:
            raise ValueError(f"{params_prefix}.num_layers must be at least 1")
        if params.num_heads < 1:
            raise ValueError(f"{params_prefix}.num_heads must be at least 1")
        if params.ffn_dim < 1:
            raise ValueError(f"{params_prefix}.ffn_dim must be at least 1")
        if not (0.0 <= params.attn_dropout <= 1.0):
            raise ValueError(f"{params_prefix}.attn_dropout must be in [0, 1]")
        if not (0.0 <= params.ffn_dropout <= 1.0):
            raise ValueError(f"{params_prefix}.ffn_dropout must be in [0, 1]")
        if params.d_model % params.num_heads != 0:
            raise ValueError(
                f"{params_prefix}.num_heads ({params.num_heads}) must divide "
                f"{params_prefix}.d_model ({params.d_model}) evenly"
            )
        if params.pooling not in {'mean', 'cls'}:
            raise ValueError(f"{params_prefix}.pooling must be one of ['mean', 'cls']")
        return

    if not isinstance(params, TCNModelParams):
        raise ValueError(f"{params_prefix} must be TCNModelParams when model.type=tcn")
    if common.sequence_length < 2:
        raise ValueError("model.common.sequence_length must be at least 2 when model.type=tcn")
    if not params.channels:
        raise ValueError(f"{params_prefix}.channels must not be empty")
    if any(int(c) < 1 for c in params.channels):
        raise ValueError(f"{params_prefix}.channels entries must be >= 1")
    if params.kernel_size < 2:
        raise ValueError(f"{params_prefix}.kernel_size must be at least 2")
    if params.dilation_base < 1:
        raise ValueError(f"{params_prefix}.dilation_base must be >= 1")
    if params.pooling not in {'last', 'mean'}:
        raise ValueError(f"{params_prefix}.pooling must be one of ['last', 'mean']")

    num_blocks = len(params.channels)
    receptive_field = 1
    for i in range(num_blocks):
        receptive_field += 2 * (params.kernel_size - 1) * (params.dilation_base ** i)

    if common.sequence_length < receptive_field:
        raise ValueError(
            "model.common.sequence_length is too short for TCN receptive field: "
            f"sequence_length={common.sequence_length}, receptive_field={receptive_field}"
        )


def _validate_model_config(model_cfg: ModelConfig):
    model_type = str(model_cfg.type).lower()
    valid_model_types = {'lstm', 'transformer', 'tcn'}
    if model_type not in valid_model_types:
        raise ValueError(f"model.type must be one of {sorted(valid_model_types)}")

    common = model_cfg.common

    expected_input_size = 19
    if common.input_size != expected_input_size:
        feature_desc = (
            "[dt, sin(n_i*dt_ik), cos(n_i*dt_ik), r_i(3), v_i(3), r_ij(3), v_ij(3), "
            "sin(u_ik), cos(u_ik), h_ik, B*_i]"
        )
        raise ValueError(
            f"model.common.input_size={common.input_size} 不合法，"
            f"当前版本必须为 {expected_input_size}。"
            f"特征向量(ECI): {feature_desc}"
        )

    if common.output_size != 3:
        logger.warning(f"Output size is {common.output_size}, expected 3 (RTN coordinates)")

    if common.sequence_length < 1:
        raise ValueError("model.common.sequence_length must be at least 1")

    if not (0.0 <= common.dropout <= 1.0):
        raise ValueError("model.common.dropout must be in [0, 1]")

    params_map = {
        'lstm': LSTMModelParams,
        'transformer': TransformerModelParams,
        'tcn': TCNModelParams,
    }

    params_by_type = getattr(model_cfg, 'params_by_type', None)
    if not isinstance(params_by_type, dict):
        raise ValueError("model.params_by_type must be present and be a mapping")
    expected_types = set(params_map.keys())
    unknown_types = sorted(set(params_by_type.keys()) - expected_types)
    if unknown_types:
        raise ValueError(f"model.params_by_type has unknown model types: {unknown_types}")
    missing_types = sorted(expected_types - set(params_by_type.keys()))
    if missing_types:
        raise ValueError(f"model.params_by_type is missing required model types: {missing_types}")

    parsed_params_by_type = {}
    for m_type in sorted(expected_types):
        raw_params = params_by_type[m_type]
        if isinstance(raw_params, params_map[m_type]):
            params_obj = raw_params
        else:
            params_obj = _instantiate_dataclass_strict(
                params_map[m_type],
                raw_params,
                f"model.params_by_type.{m_type}"
            )
        parsed_params_by_type[m_type] = params_obj

    active_params = model_cfg.params
    _validate_model_specific_params(
        model_type, common, active_params, params_prefix="model.params"
    )

    active_expected_dict = asdict(parsed_params_by_type[model_type])
    active_actual_dict = asdict(active_params) if is_dataclass(active_params) else dict(active_params)
    if active_actual_dict != active_expected_dict:
        raise ValueError(
            "model.params must exactly match model.params_by_type[model.type] "
            f"(model.type={model_type})"
        )


def validate_config(config: Config):
    """
    校验配置项

    参数：
        config: Config 对象

    异常：
        ValueError: 若配置非法
    """
    if config.experiment.seed < 0:
        raise ValueError("Seed must be non-negative")

    if config.data.split.val_days < 1:
        raise ValueError("Validation days must be at least 1")
    try:
        tle_start = datetime.strptime(config.data.tle.start_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("data.tle.start_date must be in YYYY-MM-DD format") from exc
    try:
        tle_end = datetime.strptime(config.data.tle.end_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("data.tle.end_date must be in YYYY-MM-DD format") from exc
    if tle_start > tle_end:
        raise ValueError("data.tle.start_date must be <= data.tle.end_date")
    if config.data.tle.orbit_samples_n < 1:
        raise ValueError("data.tle.orbit_samples_n must be at least 1")
    if config.data.filter.min_sat_samples_before_split < 0:
        raise ValueError("data.filter.min_sat_samples_before_split must be >= 0")
    if config.data.filter.min_train_samples < 0:
        raise ValueError("data.filter.min_train_samples must be >= 0")
    if config.data.filter.min_val_samples < 0:
        raise ValueError("data.filter.min_val_samples must be >= 0")
    if config.data.filter.min_test_samples_for_eval < 0:
        raise ValueError("data.filter.min_test_samples_for_eval must be >= 0")

    valid_feature_modes = {'t_error_stats', 'orbital_alt_inc'}
    if config.clustering.feature_mode not in valid_feature_modes:
        raise ValueError(
            f"clustering.feature_mode must be one of {sorted(valid_feature_modes)}"
        )
    valid_clustering_methods = {'agglomerative', 'gmm', 'kmeans'}
    if config.clustering.method not in valid_clustering_methods:
        raise ValueError(
            f"clustering.method must be one of {sorted(valid_clustering_methods)}"
        )
    if config.clustering.n_clusters < 1:
        raise ValueError("Number of clusters must be at least 1")
    if config.clustering.method in {'agglomerative', 'gmm'} and bool(config.clustering.balance_by_samples):
        raise ValueError("clustering.balance_by_samples must be false when clustering.method is 'gmm' or 'agglomerative'")
    if not (0.0 <= float(config.clustering.balance_tolerance) < 1.0):
        raise ValueError("clustering.balance_tolerance must be in [0, 1)")
    if not (0.0 <= float(config.clustering.balance_relax_step) < 1.0):
        raise ValueError("clustering.balance_relax_step must be in [0, 1)")
    if int(config.clustering.balance_max_relax_steps) < 0:
        raise ValueError("clustering.balance_max_relax_steps must be >= 0")

    _validate_model_config(config.model)

    if config.training.collaborative.num_rounds < 1:
        raise ValueError("Number of collaborative iterations must be at least 1")
    if config.training.batch_size < 1:
        raise ValueError("Batch size must be at least 1")
    valid_loss_types = {"loss1", "loss2", "loss3"}
    if config.training.loss_type not in valid_loss_types:
        raise ValueError(
            f"training.loss_type must be one of {sorted(valid_loss_types)}"
        )
    if config.training.loss_type in {"loss2", "loss3"} and config.training.loss_delta <= 0:
        raise ValueError(
            "training.loss_delta must be positive when training.loss_type is loss2 or loss3"
        )
    if config.training.loss_type == "loss3" and config.training.loss3_mu < 0:
        raise ValueError("training.loss3_mu must be >= 0 when training.loss_type=loss3")
    axis_weights = list(config.training.loss_axis_weights or [])
    if len(axis_weights) != 3:
        raise ValueError("training.loss_axis_weights must contain exactly 3 values for R/T/N")
    if any(w < 0 for w in axis_weights):
        raise ValueError("training.loss_axis_weights values must be >= 0")
    if sum(axis_weights) <= 0:
        raise ValueError("training.loss_axis_weights must have a positive sum")
    valid_model_selection_metrics = {"val_loss", "val_t_rms_km"}
    if config.training.model_selection.metric not in valid_model_selection_metrics:
        raise ValueError(
            f"training.model_selection.metric must be one of "
            f"{sorted(valid_model_selection_metrics)}"
        )
    valid_model_selection_modes = {"min", "max"}
    if config.training.model_selection.mode not in valid_model_selection_modes:
        raise ValueError(
            f"training.model_selection.mode must be one of "
            f"{sorted(valid_model_selection_modes)}"
        )
    if config.training.model_selection.min_delta < 0:
        raise ValueError("training.model_selection.min_delta must be >= 0")
    if config.training.validation.eval_every_rounds < 1:
        raise ValueError("training.validation.eval_every_rounds must be >= 1")
    if config.training.validation.eval_every_rounds > config.training.collaborative.num_rounds:
        raise ValueError(
            "training.validation.eval_every_rounds must be <= training.collaborative.num_rounds"
        )
    try:
        min_lr = float(config.training.lr_schedule.min_lr)
    except (TypeError, ValueError) as exc:
        raise ValueError("training.lr_schedule.min_lr must be a valid number") from exc
    if min_lr <= 0:
        raise ValueError("training.lr_schedule.min_lr must be positive")

    try:
        warmup_rounds = int(config.training.lr_schedule.warmup_rounds)
    except (TypeError, ValueError) as exc:
        raise ValueError("training.lr_schedule.warmup_rounds must be an integer") from exc
    if warmup_rounds < 0:
        raise ValueError("training.lr_schedule.warmup_rounds must be >= 0")
    if warmup_rounds >= config.training.collaborative.num_rounds:
        raise ValueError(
            "training.lr_schedule.warmup_rounds must be < training.collaborative.num_rounds"
        )

    try:
        warmup_start_factor = float(config.training.lr_schedule.warmup_start_factor)
    except (TypeError, ValueError) as exc:
        raise ValueError("training.lr_schedule.warmup_start_factor must be a valid number") from exc
    if not (0 < warmup_start_factor <= 1):
        raise ValueError("training.lr_schedule.warmup_start_factor must be in (0, 1]")

    if "your_email" in config.data.spacetrack.username.lower():
        logger.warning("⚠️  Space-Track username not configured! Please update config.yaml")
    if "your_password" in config.data.spacetrack.password.lower():
        logger.warning("⚠️  Space-Track password not configured! Please update config.yaml")

    logger.info("Configuration validation passed")
