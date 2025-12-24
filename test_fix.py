#!/usr/bin/env python
"""
测试修复是否成功的简单脚本
"""
import torch
from perturbench.modelcore.nn.mix_pert_encoder import PertAggregator

# 测试PertAggregator
pert_aggregator = PertAggregator(emb_dim=10, output_dim=20)

# 创建测试batch：3个样本，每个样本有不同数量的perturbation
test_batch = [
    [torch.randn(10), torch.randn(10)],  # 样本1：2个perturbation
    [torch.randn(10)],                   # 样本2：1个perturbation
    [torch.randn(10), torch.randn(10), torch.randn(10)]  # 样本3：3个perturbation
]

print(f"Input batch size: {len(test_batch)}")
print(f"Perturbation counts per sample: {[len(x) for x in test_batch]}")
print(f"Total perturbations: {sum(len(x) for x in test_batch)}")

try:
    output = pert_aggregator(test_batch)
    print(f"Output shape: {output.shape}")
    print("✅ PertAggregator修复成功！")
    print(f"输出维度与batch size匹配：{output.shape[0] == len(test_batch)}")
except Exception as e:
    print(f"❌ PertAggregator仍有问题：{e}")

print("\n测试完成！")
