
Hydra 初始化 `cfg` 的过程如下：

## Hydra 初始化 cfg 的机制

### 1. 装饰器的作用

```398:399:src/perturbench/modelcore/train.py
@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
```

`@hydra.main` 装饰器会：
- 拦截函数调用
- 根据 `config_path` 和 `config_name` 加载配置
- 合并所有相关配置文件
- 将合并后的 `DictConfig` 作为参数传递给被装饰的函数

### 2. 配置文件的层级结构

从 `train.yaml` 可以看到：

```5:13:src/perturbench/configs/train.yaml
defaults:
  - _self_
  - data: default
  - model: biolord
  - callbacks: default
  - logger: default
  - trainer: default
  - paths: default
  - hydra: default
```

`defaults` 列表定义了配置的合并顺序：
- `_self_`：当前文件的内容
- `data: default`：加载 `configs/data/default.yaml`
- `model: biolord`：加载 `configs/model/biolord.yaml`
- 其他类似

### 3. 配置合并流程

Hydra 按以下步骤合并配置：

1. 加载主配置：读取 `train.yaml`
2. 解析 defaults：按顺序加载每个子配置
3. 递归合并：子配置也可以有自己的 `defaults`（如 `data/default.yaml` 包含 `transform: default`）
4. 应用覆盖：命令行参数会覆盖配置文件中的值
5. 创建 DictConfig：最终合并结果转换为 `DictConfig` 对象

### 4. 实际执行流程

当调用 `main()` 时：

```441:475:src/perturbench/modelcore/train.py
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
        _CURRENT_MODEL_NAME = _PARSER_ARGS.model  # Store model name for batch_size auto-adjustment
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
```

执行顺序：
1. 解析命令行参数（第 444 行）
2. 转换为 Hydra 覆盖格式（第 450-469 行）
3. 修改 `sys.argv`（第 472 行）
4. 调用 `main()`（第 475 行）

当 `main()` 被调用时，`@hydra.main` 装饰器会：
- 读取 `sys.argv` 中的 Hydra 覆盖参数
- 加载并合并所有配置文件
- 将最终配置作为 `cfg` 参数传入 `main` 函数

### 5. 配置覆盖优先级

从代码中可以看到覆盖优先级（从高到低）：

```402:403:src/perturbench/modelcore/train.py
    # CLI parser args have highest priority: use them to override Hydra cfg
    cfg = _apply_cli_overrides(cfg, _PARSER_ARGS)
```

1. 命令行参数（通过 `_apply_cli_overrides`）
2. Hydra 命令行覆盖（如 `model=cpa`）
3. 配置文件中的值

### 总结

Hydra 通过装饰器模式自动完成：
- 配置文件发现和加载
- 多层级配置合并
- 命令行参数解析和覆盖
- 类型转换（转换为 `DictConfig`）

最终，`main(cfg: DictConfig)` 中的 `cfg` 包含了所有合并后的配置，可以直接使用，例如 `cfg.model.lr`、`cfg.data.train_batch_size` 等。