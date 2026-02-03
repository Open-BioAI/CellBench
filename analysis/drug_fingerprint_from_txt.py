#!/usr/bin/env python3
"""
根据 unique_drug_pt.txt 中的药物名，用 RDKit + PubChem 生成分子指纹。

功能：
- 读取一个纯文本文件（每行一个药物名），默认：analysis/unique_drug_pt.txt
- 使用 PubChem 查询每个药物的 canonical SMILES
    - 优先按名称查询
    - 若名称解析失败，再尝试将其当作 SMILES 解析
- 使用 RDKit 计算 Morgan fingerprint（ECFP-like）
- 将结果保存为：
    1) 一个 CSV：包含 name, cid, smiles, success, error 信息
    2) 一个 .pt 文件：包含 names, smiles, cids, fps（tensor 形状 [N, fp_size]）

依赖：
- rdkit
- pubchempy
- torch
- pandas, numpy

示例用法：

    cd /fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main

    python analysis/drug_fingerprint_from_txt.py \
        --input-txt analysis/unique_drug_pt.txt \
        --output-csv analysis/drug_fingerprints.csv \
        --output-pt analysis/drug_fingerprints.pt \
        --fp-size 2048 \
        --fp-radius 2
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# try:
#     import torch
# except ImportError:
#     torch = None  # 允许导入失败，运行时再报错

from rdkit import Chem
from rdkit.Chem import AllChem
import pubchempy as pcp


def load_names(txt_path: Path) -> List[str]:
    """从 txt 文件中读取药物名列表，自动去掉空行和重复。"""
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Input txt not found: {txt_path}")

    names: List[str] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            names.append(s)

    # 保持顺序的去重
    seen = set()
    unique_names: List[str] = []
    for n in names:
        if n not in seen:
            unique_names.append(n)
            seen.add(n)

    print(f"[INFO] 从 {txt_path} 读取到 {len(unique_names)} 个 unique 药物名")
    return unique_names


def resolve_name_to_smiles(name: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    使用 PubChem 将药物名解析为 canonical SMILES。

    返回: (smiles, cid, error_msg)
    - 如果成功: error_msg 为 None
    - 如果失败: smiles, cid 为 None，error_msg 为错误信息
    """
    # 先按名称在 PubChem 中查找
    compounds = pcp.get_compounds(name, "name")
    if compounds:
        c = compounds[0]
        # 使用新的 API：connectivity_smiles 替代 canonical_smiles
        smiles = getattr(c, 'connectivity_smiles', None) or getattr(c, 'canonical_smiles', None)
        cid = c.cid
        if smiles:
            return smiles, cid, None

    # 如果按名称找不到，尝试将 name 当作 SMILES 解析
    mol = Chem.MolFromSmiles(name)
    if mol is not None:
        smiles = Chem.MolToSmiles(mol)
        return smiles, None, None

    return None, None, f"PubChem 和 SMILES 解析均失败"


def smiles_to_morgan_fp(
    smiles: str,
    fp_size: int = 2048,
    radius: int = 2,
) -> Optional[np.ndarray]:
    """从 SMILES 生成 Morgan fingerprint，返回 shape=(fp_size,) 的 0/1 numpy 向量。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # 使用 GetMorganFingerprintAsBitVect（当前 RDKit 版本不支持 GetMorganGenerator）
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size)
    arr = np.zeros((fp_size,), dtype=np.uint8)
    # RDKit 提供的方式：将 bit vector 拷贝到 numpy 数组
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate drug fingerprints from a txt list of drug names using RDKit + PubChem."
    )
    parser.add_argument(
        "--input-txt",
        type=str,
        default="analysis/unique_drug_pt.txt",
        help="Input txt file with one drug name per line (default: analysis/unique_drug_pt.txt).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="analysis/drug_fingerprints.csv",
        help="Output CSV path (default: analysis/drug_fingerprints.csv).",
    )
    parser.add_argument(
        "--output-pt",
        type=str,
        default="analysis/drug_fingerprints.pt",
        help="Output .pt path (default: analysis/drug_fingerprints.pt).",
    )
    parser.add_argument(
        "--fp-size",
        type=int,
        default=2048,
        help="Fingerprint size (number of bits), default: 2048.",
    )
    parser.add_argument(
        "--fp-radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius, default: 2.",
    )

    args = parser.parse_args()

    input_txt = Path(args.input_txt)
    output_csv = Path(args.output_csv)
    output_pt = Path(args.output_pt)
    fp_size = args.fp_size
    fp_radius = args.fp_radius

    # if torch is None:
    #     raise ImportError("torch 未安装，无法保存 .pt 文件。请先在环境中安装 torch。")

    names = load_names(input_txt)

    records = []
    fps: List[np.ndarray] = []

    for name in names:
        print(f"[INFO] 处理药物: {name}")
        smiles, cid, err = resolve_name_to_smiles(name)

        rec = {
            "name": name,
            "cid": cid,
            "smiles": smiles,
            "success": False,
            "error": err,
        }

        fp_vec = None
        if smiles is not None:
            fp_vec = smiles_to_morgan_fp(smiles, fp_size=fp_size, radius=fp_radius)
            if fp_vec is None:
                rec["error"] = (rec["error"] + "; " if rec["error"] else "") + "RDKit 分子解析失败"
            else:
                rec["success"] = True

        records.append(rec)
        if fp_vec is not None:
            fps.append(fp_vec)
        else:
            # 对于失败的，填充全 0 向量，保证与 names 对齐
            fps.append(np.zeros((fp_size,), dtype=np.uint8))

    # 保存 CSV
    df = pd.DataFrame.from_records(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] 已保存 CSV 到: {output_csv} (共 {len(df)} 条记录，其中 success={df['success'].sum()})")

    # # 保存 .pt
    # fps_array = np.stack(fps, axis=0)  # (N, fp_size)
    # fps_tensor = torch.from_numpy(fps_array.astype(np.uint8))

    # payload = {
    #     "names": names,
    #     "smiles": df["smiles"].tolist(),
    #     "cids": df["cid"].tolist(),
    #     "success": df["success"].tolist(),
    #     "fps": fps_tensor,  # (N, fp_size), uint8 0/1
    #     "fp_size": fp_size,
    #     "fp_radius": fp_radius,
    # }
    # output_pt.parent.mkdir(parents=True, exist_ok=True)
    # torch.save(payload, output_pt)
    # print(f"[INFO] 已保存指纹到 .pt 文件: {output_pt}")


if __name__ == "__main__":
    main()


