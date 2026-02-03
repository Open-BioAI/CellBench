#!/usr/bin/env python3
"""
收集多个 h5ad 文件中 obs['drug_pt'] 的所有唯一药物名。

功能：
- 支持多个输入 h5ad 文件
- 从每个文件的 obs[drug_col] 提取字符串
- 支持用 '+' 连接的组合药物（例如 'A+B'），会拆成 'A' 和 'B'
- 去重后写入一个 txt 文件，每行一个药物名

用法示例：

    # 最简单用法：指定若干 h5ad 文件
    python collect_unique_drug_pt.py \\
        --h5ad-files \\
        ../tasks/unseen_cells/zeroshot/McFarlandTsherniak2020_stack.h5ad \\
        ../tasks/unseen_cells/zeroshot/Kang2018_CD4Tcells_stack.h5ad \\
        --output unique_drug_pt.txt

    # 如果列名不是 'drug_pt'，可以指定：
    python collect_unique_drug_pt.py \\
        --h5ad-files file1.h5ad file2.h5ad \\
        --drug-col drug \\
        --output unique_drug_names.txt
"""

import argparse
from pathlib import Path
from typing import Iterable, Set

import numpy as np
import scanpy as sc


def collect_unique_from_column(
    h5ad_paths: Iterable[Path],
    drug_col: str = "drug_pt",
    delimiter: str = "+",
) -> Set[str]:
    """
    从多个 h5ad 文件的 obs[drug_col] 中收集唯一药物名称。

    - 支持空值 / NaN / 'nan' 等情况，自动跳过
    - 支持用 delimiter 连接的组合药物，例如 'A+B'
    """
    unique_drugs: Set[str] = set()

    for path in h5ad_paths:
        path = Path(path)
        if not path.exists():
            print(f"[WARNING] h5ad 文件不存在，跳过: {path}")
            continue

        print(f"[INFO] 读取 h5ad: {path}")
        adata = sc.read_h5ad(path)
        print(f"       形状: {adata.shape}")

        if drug_col not in adata.obs.columns:
            print(f"[WARNING] obs 中不存在列 '{drug_col}'，跳过该文件")
            continue

        col = adata.obs[drug_col]
        # 转成字符串，方便 split
        values = col.astype(str).tolist()

        for raw in values:
            if raw is None:
                continue
            s = str(raw).strip()
            if s == "" or s.lower() == "nan":
                continue

            # 拆分组合药物
            parts = s.split(delimiter) if delimiter in s else [s]
            for p in parts:
                name = p.strip()
                if not name:
                    continue
                if name.lower() == "nan":
                    continue
                unique_drugs.add(name)

    return unique_drugs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect unique drug names from obs[drug_pt] across multiple h5ad files."
    )
    parser.add_argument(
        "--h5ad-files",
        type=str,
        nargs="+",
        required=True,
        help="List of input h5ad files.",
    )
    parser.add_argument(
        "--drug-col",
        type=str,
        default="drug_pt",
        help="Column name in obs containing drug identifiers (default: 'drug_pt').",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default="+",
        help="Delimiter used to join multiple drugs in one entry (default: '+').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="unique_drug_pt.txt",
        help="Output txt file path (default: 'unique_drug_pt.txt').",
    )

    args = parser.parse_args()

    h5ad_paths = [Path(p) for p in args.h5ad_files]
    output_path = Path(args.output)

    unique_drugs = collect_unique_from_column(
        h5ad_paths=h5ad_paths,
        drug_col=args.drug_col,
        delimiter=args.delimiter,
    )

    sorted_drugs = sorted(unique_drugs)
    print(f"[INFO] 共收集到 {len(sorted_drugs)} 个唯一药物名，将写入: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for name in sorted_drugs:
            f.write(f"{name}\n")

    print("[INFO] 写入完成。示例前几个药物名：")
    for name in sorted_drugs[:10]:
        print("  ", name)


if __name__ == "__main__":
    main()


