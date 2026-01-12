
import logging
import argparse
import sys
import os
import glob
from typing import List, Optional
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('spawn')
import hydra
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import Logger
from perturbench.modelcore.utils import multi_instantiate
from hydra.core.hydra_config import HydraConfig

log = logging.getLogger(__name__)


_PARSER_ARGS = None


def _str2bool(v: str | bool | None) -> bool | None:
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower() in {"true", "1", "yes", "y"}:
            return True
        if v.lower() in {"false", "0", "no", "n"}:
            return False
    return None


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Parse command line flags (e.g. from wandb sweep) and later override Hydra config.
    We deliberately keep these as standard --flags so that wandb can inject them.
    """
    parser = argparse.ArgumentParser(add_help=False)

    # config groups (will be translated into Hydra overrides like data=..., model=...)
    parser.add_argument("--data", dest="data", type=str)
    parser.add_argument("--model", dest="model", type=str)
    parser.add_argument("--logger", dest="logger", type=str)

    # leaf parameters that we want to override directly
    parser.add_argument("--data.task", dest="data_task", type=str)
    parser.add_argument("--data.data_path", dest="data_data_path", type=str)

    parser.add_argument("--train", dest="train", type=str)
    parser.add_argument("--test", dest="test", type=str)
    parser.add_argument("--test_ckpt_type", dest="test_ckpt_type", type=str)

    parser.add_argument(
        "--trainer.log_every_n_steps",
        dest="trainer_log_every_n_steps",
        type=int,
    )
    parser.add_argument("--trainer.max_epochs", dest="trainer_max_epochs", type=int)
    parser.add_argument("--trainer.min_epochs", dest="trainer_min_epochs", type=int)
    # devices is usually a Hydra list, keep it as string so Hydra can parse it
    parser.add_argument("--trainer.devices", dest="trainer_devices", type=str)
    parser.add_argument("--trainer.strategy", dest="trainer_strategy", type=str)
    parser.add_argument("--trainer.accelerator", dest="trainer_accelerator", type=str)

    parser.add_argument("--data.val_batch_size", dest="data_val_batch_size", type=int)
    parser.add_argument(
        "--data.test_batch_size",
        dest="data_test_batch_size",
        type=int,
    )
    parser.add_argument(
        "--data.transform.use_covs",
        dest="data_transform_use_covs",
        type=str,
    )

    parser.add_argument("--model.use_mask", dest="model_use_mask", type=str)
    parser.add_argument("--model.use_cell_emb", dest="model_use_cell_emb", type=str)
    parser.add_argument("--data.train_batch_size", dest="data_train_batch_size", type=int)
    parser.add_argument("--model.dropout", dest="model_dropout", type=float)
    parser.add_argument("--model.lr", dest="model_lr", type=float)
    parser.add_argument(
        "--model.lr_scheduler_max_lr",
        dest="model_lr_scheduler_max_lr",
        type=float,
    )
    parser.add_argument(
        "--model.lr_scheduler_factor",
        dest="model_lr_scheduler_factor",
        type=float,
    )
    parser.add_argument(
        "--model.lr_scheduler_mode",
        dest="model_lr_scheduler_mode",
        type=str,
    )
    parser.add_argument(
        "--model.lr_scheduler_patience",
        dest="model_lr_scheduler_patience",
        type=int,
    )
    parser.add_argument("--model.wd", dest="model_wd", type=float)

    # early stopping parameters
    parser.add_argument(
        "--callbacks.early_stopping.monitor",
        dest="callbacks_early_stopping_monitor",
        type=str,
    )
    parser.add_argument(
        "--callbacks.early_stopping.patience",
        dest="callbacks_early_stopping_patience",
        type=int,
    )

    # logger / wandb related
    parser.add_argument(
        "--logger.wandb.project",
        dest="logger_wandb_project",
        type=str,
    )
    parser.add_argument(
        "--logger.wandb.name",
        dest="logger_wandb_name",
        type=str,
    )

    return parser


def _apply_cli_overrides(cfg: DictConfig, args: argparse.Namespace | None) -> DictConfig:
    """Override Hydra cfg using highest-priority CLI arguments."""
    if args is None:
        return cfg

    # Allow adding new keys via CLI (e.g., data.task, model.lr) even when struct is enabled.
    # We only relax struct during the override phase.
    from omegaconf import OmegaConf as _OC
    _OC.set_struct(cfg, False)

    mapping: dict[str, str] = {
        # config groups are handled via Hydra overrides in __main__, not here:
        # "data": "data",
        # "model": "model",
        # "logger": "logger",
        "data_task": "data.task",
        "data_data_path": "data.data_path",
        "train": "train",
        "test": "test",
        "test_ckpt_type": "test_ckpt_type",
        "trainer_log_every_n_steps": "trainer.log_every_n_steps",
        "trainer_max_epochs": "trainer.max_epochs",
        "trainer_min_epochs": "trainer.min_epochs",
        "trainer_strategy": "trainer.strategy",
        "trainer_accelerator": "trainer.accelerator",
        # "trainer_devices" is translated to a Hydra override in __main__
        "data_val_batch_size": "data.val_batch_size",
        "data_test_batch_size": "data.test_batch_size",
        "data_transform_use_covs": "data.transform.use_covs",
        "model_use_mask": "model.use_mask",
        "model_use_cell_emb": "model.use_cell_emb",
        "data_train_batch_size": "data.train_batch_size",
        "model_dropout": "model.dropout",
        "model_lr": "model.lr",
        "model_lr_scheduler_max_lr": "model.lr_scheduler_max_lr",
        "model_lr_scheduler_factor": "model.lr_scheduler_factor",
        "model_lr_scheduler_mode": "model.lr_scheduler_mode",
        "model_lr_scheduler_patience": "model.lr_scheduler_patience",
        "model_wd": "model.wd",
        "callbacks_early_stopping_monitor": "callbacks.early_stopping.monitor",
        "callbacks_early_stopping_patience": "callbacks.early_stopping.patience",
        "logger_wandb_project": "logger.wandb.project",
        "logger_wandb_name": "logger.wandb.name",
    }

    for attr, key in mapping.items():
        if not hasattr(args, attr):
            continue
        value = getattr(args, attr)
        if value is None:
            continue

        # Convert string booleans for known boolean fields
        if key in {
            "train",
            "test",
            "data.transform.use_covs",
            "model.use_mask",
            "model.use_cell_emb",
        }:
            value = _str2bool(value)
            if value is None:
                continue
        OmegaConf.update(cfg, key, value, merge=False)

    # Allow dynamic batch_size adjustment for all models (including state_sm)
    # Removed forced batch_size setting - models should handle batch_size requirements internally if needed

    return cfg


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    查找最新的 checkpoint 文件，用于自动恢复训练。
    
    查找顺序：
    1. 先查找 last.ckpt（如果存在）
    2. 然后查找 checkpoints/ 下最新的 checkpoint
    3. 返回最新的 checkpoint 路径
    
    Args:
        output_dir: Hydra 输出目录（包含 checkpoints 子目录）
        
    Returns:
        最新的 checkpoint 路径，如果不存在则返回 None
    """
    if not output_dir or not os.path.exists(output_dir):
        return None
    
    checkpoints = []
    
    # 1. 查找 last.ckpt（最优先）
    last_ckpt = os.path.join(output_dir, "last.ckpt")
    if os.path.exists(last_ckpt):
        log.info(f"Found last.ckpt: {last_ckpt}")
        return last_ckpt
    
    # 2. 查找 checkpoints/ 下的所有 checkpoint（现在只有一个统一的目录）
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    if os.path.exists(ckpt_dir):
        ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        checkpoints.extend(ckpts)
    
    # 4. 查找其他可能的 checkpoint 位置
    for pattern in ["*.ckpt", "checkpoints/**/*.ckpt"]:
        found = glob.glob(os.path.join(output_dir, pattern), recursive=True)
        checkpoints.extend(found)
    
    if not checkpoints:
        return None
    
    # 按修改时间排序，返回最新的
    latest_ckpt = max(checkpoints, key=os.path.getmtime)
    log.info(f"Found latest checkpoint: {latest_ckpt} (modified: {os.path.getmtime(latest_ckpt)})")
    return latest_ckpt


def get_auto_experiment_name(cfg: DictConfig, model_name: str) -> str:
    OmegaConf.set_struct(cfg, False)
    # 生成包含模型名、学习率、权重衰减和批大小的 run name
    lr = cfg.get("model", {}).get("lr", "unknown")
    wd = cfg.get("model", {}).get("wd", "unknown")
    batch_size = cfg.get("data", {}).get("train_batch_size", "unknown")

    # 格式化参数（避免科学计数法）
    if isinstance(lr, float):
        lr_str = f"{lr:.0e}" if lr < 0.01 else f"{lr}"
    else:
        lr_str = str(lr)

    if isinstance(wd, float):
        wd_str = f"{wd:.0e}" if wd < 0.01 else f"{wd}"
    else:
        wd_str = str(wd)

    batch_size_str = str(batch_size)

    auto_name = f"{model_name}_lr{lr_str}_wd{wd_str}_bs{batch_size_str}"

    # 修改配置（在 logger 创建之前）
    OmegaConf.update(cfg, "logger.wandb.name", auto_name, merge=False)
    log.info("Auto-generated wandb run name: %s (model: %s, lr: %s, wd: %s, bs: %s)",
             auto_name, model_name, lr_str, wd_str, batch_size_str)
    return auto_name

def train(runtime_context: dict):

    cfg = runtime_context["cfg"]
    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Set sample_mode and cell_set_len based on model type
    # Model decides data packaging mode: cell-wise [B,G] vs set-wise [B,S,G]
    model_target = cfg.model.get("_target_", "")
    is_state_transition = "StateTransitionPerturbationModel" in model_target or "state_transition" in model_target.lower()
    
    # Set sample_mode: "set" for state_transition models, "cell" for others
    sample_mode = "set" if is_state_transition else "cell"
    OmegaConf.update(cfg, "data.sample_mode", sample_mode, merge=False)
    
    # Set cell_set_len: 128 (or from model config) for state_transition, None for others
    cell_set_len = cfg.model.get("cell_set_len", 128) if is_state_transition else None
    OmegaConf.update(cfg, "data.cell_set_len", cell_set_len, merge=False)
    
    log.info(f"Model {model_target}: sample_mode={sample_mode}, cell_set_len={cell_set_len}")

    log.info("Instantiating datamodule <%s>", cfg.data._target_)
    datamodule: L.LightningDataModule =hydra.utils.instantiate(
        cfg.data,
        seed=cfg.seed
    ) # 初始化cfg.data['_target_']对应的类，并返回一个实例

    log.info("Instantiating model <%s>", cfg.model._target_)
    model = hydra.utils.instantiate(cfg.model, datamodule=datamodule) # 初始化cfg.model['_target_']对应的类，并返回一个实例

    log.info("Instantiating callbacks...")
    callbacks: List[L.Callback] = multi_instantiate(cfg.get("callbacks"))

    # 在创建 loggers 之前，先提取 model 名称并修改配置
    # 这样 WandbLogger 初始化时就能使用正确的 name
    
    # 从 cfg 中推断 model 名称（从 _target_ 提取）
    model_target = cfg.get("model", {}).get("_target_", "")
    model_name = model_target.split('.')[-1].lower()
    
    # 如果 logger.wandb.name 是默认的 "gears" 或未设置，在创建前修改配置
    auto_name = get_auto_experiment_name(cfg, model_name)
    log.info("Instantiating loggers...")
    # 尝试实例化 loggers，如果 wandb 初始化失败，继续使用其他 loggers

    loggers: List[Logger] = multi_instantiate(cfg.get("logger"))
    
    # 双重保险：如果 logger 已经创建但 name 还是 gears，再次设置
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            logger.experiment.name = auto_name
            log.info("Updated wandb run name to: %s", auto_name)
    
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers
    )

    if cfg.get("train"):
        # 自动检测并恢复 checkpoint（如果存在且未手动指定 ckpt_path）
        ckpt_path = cfg.get("ckpt_path")
        if ckpt_path is None:
            # 尝试自动查找最新的 checkpoint
            output_dir = cfg.get("paths", {}).get("output_dir", None)
            if output_dir:
                auto_ckpt = find_latest_checkpoint(output_dir)
                if auto_ckpt:
                    ckpt_path = auto_ckpt
                    log.info(f"Auto-resuming from checkpoint: {ckpt_path}")
                else:
                    log.info("No checkpoint found. Starting fresh training.")
            else:
                log.info("output_dir not found in config. Starting fresh training.")
        else:
            log.info(f"Using manually specified checkpoint: {ckpt_path}")
        
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    train_metrics = trainer.callback_metrics

    summary_metrics_dict = {}
    if cfg.get("test"):
        log.info("Starting testing!")
        # Track which checkpoint type is actually used for testing so that
        # prediction file names can include this as a suffix.
        selected_ckpt_type = "unknown"
        if cfg.get("train"):
            # if (
            #     trainer.checkpoint_callback is None
            #     or trainer.checkpoint_callback.best_model_path == ""
            # ):
            #     ckpt_path = None
            # else:
            #     ckpt_path = "best"
            # Support multiple checkpoint callbacks (e.g., loss and PCC)
            test_ckpt_type = cfg.get("test_ckpt_type", "loss")  # "loss" or "pcc"
            # Find the appropriate checkpoint callback
            ckpt_path = None
            if callbacks:
                for callback in callbacks:
                    if isinstance(callback, L.pytorch.callbacks.ModelCheckpoint):
                        # Check if this is the checkpoint we want
                        if test_ckpt_type == "loss" and "loss" in str(callback.monitor).lower():
                            if callback.best_model_path and callback.best_model_path != "":
                                ckpt_path = callback.best_model_path
                                selected_ckpt_type = "loss"
                                log.info(f"Using checkpoint from {test_ckpt_type} callback: {ckpt_path}")
                                break
                        elif test_ckpt_type == "pcc" and "pcc" in str(callback.monitor).lower():
                            if callback.best_model_path and callback.best_model_path != "":
                                ckpt_path = callback.best_model_path
                                selected_ckpt_type = "pcc"
                                log.info(f"Using checkpoint from {test_ckpt_type} callback: {ckpt_path}")
                                break
                
                # Fallback to first checkpoint callback if specific one not found
                if ckpt_path is None:
                    for callback in callbacks:
                        if isinstance(callback, L.pytorch.callbacks.ModelCheckpoint):
                            if callback.best_model_path and callback.best_model_path != "":
                                ckpt_path = callback.best_model_path
                                # Fallback checkpoint type when a specific monitor is not found
                                if test_ckpt_type in ["loss", "pcc"]:
                                    selected_ckpt_type = test_ckpt_type
                                else:
                                    selected_ckpt_type = "unknown"
                                log.info(f"Fallback: Using checkpoint: {ckpt_path}")
                                break
        else:
            ckpt_path = cfg.get("ckpt_path")

        # Pass the actually used checkpoint type to the model so that it can
        # append the suffix (e.g., predictions_loss.h5ad, predictions_pcc.h5ad).
        try:
            setattr(model, "current_test_ckpt_type", selected_ckpt_type)
        except Exception:
            # If anything goes wrong here, fall back to default behaviour
            # in which predictions are saved without a ckpt-type suffix.
            pass

        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        summary_metrics_dict = model.summary_metrics.to_dict()[
            model.summary_metrics.columns[0]
        ]

    test_metrics = trainer.callback_metrics
    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics, **summary_metrics_dict}

    return metric_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    global _PARSER_ARGS

    # CLI parser args have highest priority: use them to override Hydra cfg
    cfg = _apply_cli_overrides(cfg, _PARSER_ARGS)
    
    # 设置 TMPDIR：优先使用已设置的环境变量，否则设置为日志目录下的临时目录
    # 这可以确保临时文件创建在与 checkpoint 相同的文件系统上，避免跨设备移动问题
    import os
    existing_tmpdir = os.environ.get("TMPDIR")
    if existing_tmpdir:
        # 如果已经设置了TMPDIR（比如从run_multi_gpu_agents.sh），使用现有的设置
        log.info(f"Using existing TMPDIR: {existing_tmpdir}")
    else:
        # 否则设置到日志目录下以避免跨设备checkpoint保存问题
        log_dir = cfg.get("paths", {}).get("log_dir", None)
        if log_dir:
            # 确保 log_dir 存在
            os.makedirs(log_dir, exist_ok=True)
            # 在 log_dir 下创建临时目录
            tmp_dir = os.path.join(log_dir, ".tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            # 设置 TMPDIR 环境变量（仅对当前进程有效）
            os.environ["TMPDIR"] = tmp_dir
            log.info(f"Set TMPDIR to {tmp_dir} to avoid cross-device checkpoint save errors")

    runtime_context = {"cfg": cfg, "trial_number": HydraConfig.get().job.get("num")}

    try:
    # Train the model
        global metric_dict
        metric_dict = train(runtime_context)

        # Combined metric
        metrics_use = cfg.get("metrics_to_optimize")
        if metrics_use:
            combined_metric = sum(
                [metric_dict.get(metric) * weight for metric, weight in metrics_use.items()]
            )
            return combined_metric
    except Exception as e:
        # 捕获所有异常，记录错误信息，然后重新抛出
        # 这样 wandb agent 可以记录失败并继续下一个 run
        log.error("Training failed with exception: %s", e, exc_info=True)
        # 重新抛出异常，让 wandb agent 知道这个 run 失败了
        raise


if __name__ == "__main__":
    # 1) First parse standard CLI flags (e.g. from wandb sweep), keep unknowns
    parser = _build_arg_parser()
    _PARSER_ARGS, remaining = parser.parse_known_args()

    # 2) Translate certain argparse flags into Hydra-style overrides so that
    #    config groups (data=..., model=..., logger=...) and complex types
    #    (like trainer.devices=[0]) are handled by Hydra, while leaf parameters
    #    remain as --key flags for our own override logic.
    hydra_overrides: list[str] = []

    if getattr(_PARSER_ARGS, "data", None) is not None:
        hydra_overrides.append(f"data={_PARSER_ARGS.data}")
        _PARSER_ARGS.data = None  # avoid double-handling

    if getattr(_PARSER_ARGS, "model", None) is not None:
        hydra_overrides.append(f"model={_PARSER_ARGS.model}")
        _PARSER_ARGS.model = None

    if getattr(_PARSER_ARGS, "logger", None) is not None:
        hydra_overrides.append(f"logger={_PARSER_ARGS.logger}")
        _PARSER_ARGS.logger = None

    # Handle trainer.devices specially so that Hydra, not Lightning, parses
    # the list syntax (e.g., [0]) correctly.
    if getattr(_PARSER_ARGS, "trainer_devices", None) is not None:
        hydra_overrides.append(f"trainer.devices={_PARSER_ARGS.trainer_devices}")
        _PARSER_ARGS.trainer_devices = None

    # 3) Let Hydra see its overrides plus the remaining arguments
    sys.argv = [sys.argv[0]] + hydra_overrides + remaining

    # 4) Run Hydra entrypoint as usual; inside main we will override cfg with _PARSER_ARGS
    main()
