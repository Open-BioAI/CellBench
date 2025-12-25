
import logging
import argparse
import sys
import os
import glob
from typing import List, Optional

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
    parser.add_argument("--model.wd", dest="model_wd", type=float)

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
        "model_wd": "model.wd",
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

    return cfg


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    查找最新的 checkpoint 文件，用于自动恢复训练。
    
    查找顺序：
    1. 先查找 last.ckpt（如果存在）
    2. 然后查找 checkpoints/loss/ 和 checkpoints/pcc/ 下最新的 checkpoint
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
    
    # 2. 查找 checkpoints/loss/ 下的所有 checkpoint
    loss_ckpt_dir = os.path.join(output_dir, "checkpoints", "loss")
    if os.path.exists(loss_ckpt_dir):
        loss_ckpts = glob.glob(os.path.join(loss_ckpt_dir, "*.ckpt"))
        checkpoints.extend(loss_ckpts)
    
    # 3. 查找 checkpoints/pcc/ 下的所有 checkpoint
    pcc_ckpt_dir = os.path.join(output_dir, "checkpoints", "pcc")
    if os.path.exists(pcc_ckpt_dir):
        pcc_ckpts = glob.glob(os.path.join(pcc_ckpt_dir, "*.ckpt"))
        checkpoints.extend(pcc_ckpts)
    
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


def train(runtime_context: dict):

    cfg = runtime_context["cfg"]
    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

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
    model_name = "unknown"
    model_target = cfg.get("model", {}).get("_target_", "")
    if model_target:
        # 例如 "perturbench.modelcore.models.gears.GEARS" -> "gears"
        # 例如 "perturbench.modelcore.models.BiolordStar" -> "biolord"
        parts = model_target.split(".")
        for i, part in enumerate(parts):
            if part == "models" and i + 1 < len(parts):
                # 提取模型名，去掉可能的类名后缀（如 Star, Model, Net 等）
                raw_name = parts[i + 1]
                # 转换为小写，并去掉常见的类名后缀
                model_name = raw_name.lower()
                # 去掉常见后缀
                for suffix in ["star", "model", "net", "module", "class"]:
                    if model_name.endswith(suffix) and len(model_name) > len(suffix):
                        model_name = model_name[:-len(suffix)]
                        break
                break
        if model_name == "unknown":
            # 回退：取最后一个部分并转小写，去掉后缀
            raw_name = parts[-1].lower() if parts else "unknown"
            if raw_name != "unknown":
                model_name = raw_name
                for suffix in ["star", "model", "net", "module", "class"]:
                    if model_name.endswith(suffix) and len(model_name) > len(suffix):
                        model_name = model_name[:-len(suffix)]
                        break
    
    # 如果还是没找到，尝试从 Hydra overrides 中获取（安全方式）
    if model_name == "unknown":
        try:
            hydra_cfg = HydraConfig.get()
            # 尝试不同的路径来获取 overrides
            overrides = None
            if hasattr(hydra_cfg.job, "overrides"):
                if hasattr(hydra_cfg.job.overrides, "task"):
                    overrides = hydra_cfg.job.overrides.task
                elif isinstance(hydra_cfg.job.overrides, list):
                    overrides = hydra_cfg.job.overrides
            
            if overrides:
                for override in overrides:
                    if isinstance(override, str) and override.startswith("model="):
                        model_name = override.split("=", 1)[1]
                        break
        except Exception:
            # 如果获取 overrides 失败，忽略，使用默认的 "unknown"
            pass
    
    # 如果 logger.wandb.name 是默认的 "gears" 或未设置，在创建前修改配置
    try:
        OmegaConf.set_struct(cfg, False)
        if cfg.get("logger") and hasattr(cfg.logger, "wandb"):
            current_name = cfg.logger.wandb.get("name", None)
            if current_name is None or current_name == "gears":
                # 生成包含模型名和学习率的 run name
                lr = cfg.get("model", {}).get("lr", "unknown")
                # 格式化学习率（避免科学计数法）
                if isinstance(lr, float):
                    lr_str = f"{lr:.0e}" if lr < 0.01 else f"{lr}"
                else:
                    lr_str = str(lr)
                auto_name = f"{model_name}_lr{lr_str}"
                
                # 修改配置（在 logger 创建之前）
                OmegaConf.update(cfg, "logger.wandb.name", auto_name, merge=False)
                log.info("Auto-generated wandb run name: %s (model: %s, lr: %s)", auto_name, model_name, lr_str)
    except Exception as e:
        log.warning("Failed to update logger.wandb.name in config: %s", e)

    log.info("Instantiating loggers...")
    # 尝试实例化 loggers，如果 wandb 初始化失败，继续使用其他 loggers
    try:
        loggers: List[Logger] = multi_instantiate(cfg.get("logger"))
    except Exception as e:
        log.warning("Failed to instantiate some loggers: %s. Continuing with available loggers.", e)
        # 如果所有 loggers 都失败，尝试只实例化非 wandb loggers
        logger_cfg = cfg.get("logger", {})
        if logger_cfg:
            loggers = []
            for logger_name, logger_conf in logger_cfg.items():
                try:
                    if isinstance(logger_conf, DictConfig) and "_target_" in logger_conf:
                        target = logger_conf.get("_target_", "")
                        # 如果 wandb 失败，跳过它，继续使用其他 loggers
                        if "wandb" in target.lower():
                            log.warning("Skipping wandb logger due to initialization error. Continuing with other loggers.")
                            continue
                        loggers.append(hydra.utils.instantiate(logger_conf, _recursive_=False))
                except Exception as logger_e:
                    log.warning("Failed to instantiate logger %s: %s", logger_name, logger_e)
            if not loggers:
                log.error("All loggers failed to initialize. Training may continue without logging.")
                loggers = []
        else:
            loggers = []

    # 双重保险：如果 logger 已经创建但 name 还是 gears，再次设置
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            current_name = getattr(logger, "name", None) or getattr(logger, "_name", None)
            if current_name == "gears":
                # 重新生成 name
                lr = cfg.get("model", {}).get("lr", "unknown")
                if isinstance(lr, float):
                    lr_str = f"{lr:.0e}" if lr < 0.01 else f"{lr}"
                else:
                    lr_str = str(lr)
                auto_name = f"{model_name}_lr{lr_str}"
                
                # 尝试多种方式设置 name
                try:
                    logger.name = auto_name
                except Exception:
                    pass
                try:
                    if hasattr(logger, "experiment") and logger.experiment:
                        logger.experiment.name = auto_name
                except Exception:
                    pass
                
                log.info("Updated wandb run name to: %s", auto_name)
    
    # Auto-adjust trainer config for single GPU: disable DDP and use FP32 precision
    # This fixes the "No inf checks were recorded" error when using AMP with DDP on single GPU
    trainer_cfg = cfg.get("trainer", {})
    devices = trainer_cfg.get("devices", None)
    
    # Check if devices is 1 (single GPU)
    if devices is not None:
        if isinstance(devices, (list, tuple)):
            num_devices = len(devices)
        elif isinstance(devices, (int, str)):
            try:
                num_devices = int(devices)
            except (ValueError, TypeError):
                num_devices = 1
        else:
            num_devices = 1
        
        if num_devices == 1:
            # Single GPU: disable DDP strategy and use FP32 to avoid AMP issues
            strategy = trainer_cfg.get("strategy", None)
            if strategy and "ddp" in str(strategy).lower():
                log.warning("Single GPU detected (devices=1). Disabling DDP strategy to avoid compatibility issues.")
                OmegaConf.update(cfg, "trainer.strategy", "auto", merge=False)
            
            # Also disable mixed precision for single GPU to avoid AMP scaler issues
            precision = trainer_cfg.get("precision", None)
            if precision == 16 or precision == "16" or precision == "16-mixed":
                log.warning("Single GPU detected (devices=1). Disabling mixed precision (FP16) to avoid AMP scaler issues.")
                OmegaConf.update(cfg, "trainer.precision", 32, merge=False)
    
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
    
    # 设置 TMPDIR 到日志目录所在的文件系统，避免跨设备移动 checkpoint 时出错
    # 这样可以确保临时文件创建在与 checkpoint 相同的文件系统上
    import os
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
