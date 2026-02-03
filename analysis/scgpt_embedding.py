#!/usr/bin/env python3
"""
简化的scGPT嵌入生成脚本（相对路径版）
- 直接使用adata.var.index作为基因名
- 不进行Ensembl ID映射
- 不区分物种
- 直接使用原始count矩阵进行embedding

默认路径假设当前工作目录为 AIVC：
- 数据目录: ./data
- 模型目录: ./data/scGPT_human
- 输出目录: ./data/scgpt_embedding
可通过命令行参数覆盖。
"""

import os
import sys
import warnings
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats
from sklearn.preprocessing import StandardScaler


def check_dependencies():
    """检查必要的依赖包"""
    required_packages = [
        'numpy', 'pandas', 'scipy', 'scanpy', 'sklearn', 'scgpt'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"错误: 缺少以下依赖包: {', '.join(missing_packages)}")
        print(f"请安装: pip install {' '.join(missing_packages)}")
        return False
    
    return True


def suppress_warnings():
    """抑制常见警告"""
    warnings.filterwarnings("ignore", category=UserWarning, module="scgpt")
    warnings.filterwarnings("ignore", category=FutureWarning, module="legacy_api_wrap")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # 抑制torchtext弃用警告
    try:
        import torchtext
        torchtext.disable_torchtext_deprecation_warning()
    except ImportError:
        pass
    
    # 抑制tqdm警告
    try:
        from tqdm import TqdmWarning
        warnings.filterwarnings("ignore", category=TqdmWarning)
    except ImportError:
        pass


def check_gene_format(adata):
    """检查基因名格式"""
    print("=== 基因名格式检查 ===")
    print(f"总基因数: {len(adata.var.index)}")
    
    gene_names = adata.var.index.astype(str).tolist()[:10]
    print("前10个基因名:")
    for i, gene in enumerate(gene_names, 1):
        print(f"  {i:2d}. {gene}")
    
    ensembl_count = sum(1 for gene in gene_names if gene.startswith(('ENSG', 'ENSMUSG')))
    print(f"\nEnsembl ID检测:")
    print(f"  前10个基因中Ensembl ID数量: {ensembl_count}")
    print("==================")
    return gene_names


def process_raw_dataset(adata, model_dir, batch_size=128):
    """
    处理原始数据集生成scGPT嵌入
    - 仅在与模型词表取交集后的矩阵上，过滤“全0细胞/基因”
    - 原始 adata 不做修改；embedding 计算完后按细胞名对齐回填（未参与=NaN）
    """
    print(f"\n处理原始数据集...")
    print(f"数据形状: {adata.shape}")

    import json
    from scipy import sparse
    import scgpt as scg

    # ---------- 基本准备 ----------
    adata_copy = adata.copy()
    adata_copy.var.index = adata_copy.var.index.astype(str).str.strip()
    adata_copy.var_names_make_unique()

    # 原始计数优先
    from scipy import sparse as _sp
    if 'counts' in adata_copy.layers:
        print("使用 adata.layers['counts'] 作为原始计数数据")
        adata_copy.X = adata_copy.layers['counts']
    elif adata_copy.raw is not None:
        print("使用 adata.raw.X 作为原始计数数据")
        adata_copy.X = adata_copy.raw.X
    else:
        print("使用 adata.X 作为原始计数数据")

    if not _sp.issparse(adata_copy.X):
        adata_copy.X = _sp.csr_matrix(adata_copy.X)
    else:
        adata_copy.X = adata_copy.X.tocsr()

    check_gene_format(adata_copy)

    # ---------- 与 scGPT 词表取交集 ----------
    vocab_path = None
    for fn in ["vocab.json", "gene_vocab.json", "gene_tokenizer.json"]:
        p = Path(model_dir) / fn
        if p.exists():
            vocab_path = p
            break
    if vocab_path is None:
        print("错误: 未找到 scGPT 词表文件（vocab.json/gene_vocab.json/gene_tokenizer.json）")
        return None

    with open(vocab_path) as f:
        vocab = json.load(f)
    vocab_genes = set(vocab.keys())

    keep_genes = [g for g in adata_copy.var_names if g in vocab_genes]
    if len(keep_genes) == 0:
        print("错误: 模型词表与数据无交集基因，检查物种或基因命名（symbol/ENSG）")
        return None

    adata_sub = adata_copy[:, keep_genes].copy()
    print(f"=== 数据质量检查（在词表交集基因上）===")
    print(f"细胞数: {adata_sub.n_obs} | 交集基因数: {adata_sub.n_vars}")

    # ---------- 过滤全0细胞/基因 ----------
    X = adata_sub.X
    if sparse.issparse(X):
        nz_cells = np.asarray(X.getnnz(axis=1)).ravel()
    else:
        nz_cells = (X > 0).sum(axis=1)
    mask_cells = nz_cells > 0
    removed_cells = (~mask_cells).sum()
    if removed_cells:
        print(f"- 将移除在交集基因上全0的细胞: {removed_cells}")
    adata_in = adata_sub[mask_cells].copy()

    X2 = adata_in.X
    if sparse.issparse(X2):
        nz_genes = np.asarray(X2.getnnz(axis=0)).ravel()
    else:
        nz_genes = (X2 > 0).sum(axis=0)
    mask_genes = nz_genes > 0
    removed_genes = (~mask_genes).sum()
    if removed_genes:
        print(f"- 将移除在当前细胞中全0的基因: {removed_genes}")
    adata_in = adata_in[:, mask_genes].copy()

    print(f"过滤后细胞数: {adata_in.n_obs} | 过滤后基因数: {adata_in.n_vars}")
    if adata_in.n_obs == 0 or adata_in.n_vars == 0:
        print("错误: 过滤后为空，无法生成嵌入；请检查物种或基因命名。")
        return None

    # ---------- 生成 scGPT 嵌入 ----------
    print(f"\n=== 生成scGPT嵌入 ===")
    print(f"模型目录: {model_dir}")
    print(f"批次大小: {batch_size}")

    try:
        print("正在生成嵌入...")
        try:
            adata_emb = scg.tasks.embed_data(
                adata_in,
                model_dir=str(model_dir),
                gene_col='index',
                batch_size=batch_size,
                return_new_adata=True,
                keep_first_n_tokens=0,
            )
        except TypeError:
            adata_emb = scg.tasks.embed_data(
                adata_in,
                model_dir=str(model_dir),
                gene_col='index',
                batch_size=batch_size,
                return_new_adata=True,
            )

        if hasattr(adata_emb, 'obsm') and 'scgpt_embeddings' in adata_emb.obsm:
            emb = adata_emb.obsm['scgpt_embeddings']
        elif hasattr(adata_emb, 'X'):
            emb = adata_emb.X
        else:
            print("警告: 无法在返回对象中找到嵌入数据")
            return None

        print(f"✓ 嵌入生成成功，形状: {emb.shape}")

        # 回填到原始 adata（未参与细胞=NaN）
        d = emb.shape[1]
        full = np.full((adata.n_obs, d), np.nan, dtype=emb.dtype)

        idx_in = pd.Index(adata.obs_names).get_indexer(adata_in.obs_names)
        full[idx_in] = emb

        adata.obsm['scgpt_embeddings'] = full
        used = np.zeros(adata.n_obs, dtype=bool)
        used[idx_in] = True
        adata.obs['used_for_scgpt'] = used

        print(f"✓ 嵌入已回填到原始数据：{adata.obsm['scgpt_embeddings'].shape}（未参与细胞为 NaN）")

    except Exception as e:
        print(f"错误: 生成嵌入时出错 - {e}")
        print("跳过该数据集的嵌入生成")
        import traceback
        traceback.print_exc()
        return None

    print("================")
    return adata


def main():
    """主函数"""
    if not check_dependencies():
        sys.exit(1)
    
    try:
        import scgpt as scg  # noqa: F401
    except ImportError:
        print("错误: 无法导入scgpt")
        sys.exit(1)
    
    suppress_warnings()
    
    # 参数解析
    parser = argparse.ArgumentParser(description='生成scGPT嵌入')
    parser.add_argument('--input_h5ad', required=True,
                        help='输入的.h5ad文件路径')
    parser.add_argument('--output_h5ad', required=True,
                        help='输出的.h5ad文件路径（包含嵌入）')
    parser.add_argument('--model_dir',
                        default='/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/data/scGPT_human',
                        help='scGPT模型目录路径')

    args = parser.parse_args()

    input_file = args.input_h5ad
    output_file = args.output_h5ad
    model_dir = Path(args.model_dir)
    
    if not os.path.exists(model_dir):
        print(f"错误: 模型目录不存在: {model_dir}")
        print("请下载预训练的scGPT模型至上述目录")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"模型目录: {model_dir}")
    
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        sys.exit(1)
    
    try:
        print(f"\n从以下位置读取原始数据: {input_file}")
        adata = sc.read_h5ad(input_file)
        print(f"已加载原始数据: {adata.shape}")
        
        adata_with_embeddings = process_raw_dataset(
            adata,
            model_dir,
            batch_size=128,
        )
        
        if adata_with_embeddings is None:
            print(f"跳过保存 {os.path.basename(input_file)} - 嵌入生成失败")
            return
        
        print(f"保存带嵌入的数据到: {output_file}")
        adata_with_embeddings.write_h5ad(output_file)
        print(f"保存成功!")
        
        print(f"\n=== 验证保存的文件 ===")
        if os.path.exists(output_file):
            print(f"✓ 文件已保存: {output_file}")
            file_size = os.path.getsize(output_file) / (1024 * 1024)
            print(f"  文件大小: {file_size:.2f} MB")
            try:
                test_adata = sc.read_h5ad(output_file)
                if 'scgpt_embeddings' in test_adata.obsm:
                    print(f"✓ 嵌入已保存到文件中的obsm['scgpt_embeddings']")
                    print(f"  嵌入形状: {test_adata.obsm['scgpt_embeddings'].shape}")
                else:
                    print("✗ 警告: 文件中未找到嵌入数据")
            except Exception as e:
                print(f"✗ 无法验证文件内容: {e}")
        else:
            print(f"✗ 文件保存失败: {output_file}")
        print("==================")
        
    except Exception as e:
        print(f"错误: 处理数据集时出错 - {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


