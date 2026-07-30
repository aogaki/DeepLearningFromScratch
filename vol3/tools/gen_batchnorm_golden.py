"""BatchNorm2d の per-op パリティ golden を生成する(vol3 拡張・UNet 前準備)。

事前登録済みの容疑者リスト(docs/framework-v1-notes.md)を一発検出するための最小 golden:
  1. 正規化に使う分散 = biased(÷N)、running_var への蓄積 = unbiased(÷(N−1))
  2. momentum の向き: running = (1−m)·running + m·batch、m=0.1
  3. eps=1e-5 は √ の内側: √(var + eps)
  4. 統計の軸: チャネルごとに N·H·W をまとめて平均・分散

内容(すべて f32、PyTorch デフォルト設定の BatchNorm2d(C=3)):
  - gamma/beta は非自明な値(1/0 のデフォルトではスケール・シフトの適用ミスに盲目)
  - train step1: y1・backward(gx1/dgamma1/dbeta1)・更新後 running stats(rm1/rv1)
  - train step2: y2・running stats の漸化式 2 周目(rm2/rv2)
  - eval: y3_eval(蓄積した running stats で正規化)
  - 診断用: x1 のチャネル別バッチ統計(biased/unbiased 両方)

実行: venv-torch などで `python gen_batchnorm_golden.py`
出力: vol3/dataset/batchnorm_golden.npz
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
N, C, H, W = 4, 3, 2, 3

bn = nn.BatchNorm2d(C)  # eps=1e-5, momentum=0.1, affine=True
gamma = torch.tensor([0.5, 1.5, -0.8])
beta = torch.tensor([0.1, -0.2, 0.3])
with torch.no_grad():
    bn.weight.copy_(gamma)
    bn.bias.copy_(beta)

x1 = torch.randn(N, C, H, W)
x2 = torch.randn(N, C, H, W)
x3 = torch.randn(N, C, H, W)
gy1 = torch.randn(N, C, H, W)

# --- train step 1: forward + backward ---
bn.train()
x1.requires_grad_(True)
y1 = bn(x1)
y1.backward(gy1)
gx1 = x1.grad.detach().clone()
dgamma1 = bn.weight.grad.detach().clone()
dbeta1 = bn.bias.grad.detach().clone()
rm1 = bn.running_mean.detach().clone()
rv1 = bn.running_var.detach().clone()

# --- train step 2: forward のみ(running stats の漸化式 2 周目) ---
# no_grad でも train モードなら running stats は更新される
with torch.no_grad():
    y2 = bn(x2)
rm2 = bn.running_mean.detach().clone()
rv2 = bn.running_var.detach().clone()

# --- eval: 蓄積した running stats で正規化 ---
bn.eval()
with torch.no_grad():
    y3_eval = bn(x3)

# --- 診断用: x1 のチャネル別バッチ統計 ---
xm = x1.detach().transpose(0, 1).reshape(C, -1)  # (C, N*H*W)
batch_mean1 = xm.mean(dim=1)
batch_var_biased1 = xm.var(dim=1, unbiased=False)
batch_var_unbiased1 = xm.var(dim=1, unbiased=True)

out_path = Path(__file__).resolve().parent.parent / "dataset" / "batchnorm_golden.npz"
np.savez(
    out_path,
    x1=x1.detach().numpy(),
    x2=x2.numpy(),
    x3=x3.numpy(),
    gy1=gy1.numpy(),
    gamma=gamma.numpy(),
    beta=beta.numpy(),
    y1=y1.detach().numpy(),
    y2=y2.numpy(),
    y3_eval=y3_eval.numpy(),
    gx1=gx1.numpy(),
    dgamma1=dgamma1.numpy(),
    dbeta1=dbeta1.numpy(),
    rm1=rm1.numpy(),
    rv1=rv1.numpy(),
    rm2=rm2.numpy(),
    rv2=rv2.numpy(),
    batch_mean1=batch_mean1.numpy(),
    batch_var_biased1=batch_var_biased1.numpy(),
    batch_var_unbiased1=batch_var_unbiased1.numpy(),
)
print(f"saved: {out_path}")

# 容疑者の存在証明(人間用の答え合わせ)
print("\n--- 容疑者 1・2 の実地確認 ---")
print("batch_mean1          :", batch_mean1.numpy())
print("rm1 (= 0.1*batch_mean):", rm1.numpy())
print("batch_var_biased1    :", batch_var_biased1.numpy())
print("batch_var_unbiased1  :", batch_var_unbiased1.numpy())
print("rv1 (= 0.9*1 + 0.1*batch_var_UNBIASED):", rv1.numpy())
