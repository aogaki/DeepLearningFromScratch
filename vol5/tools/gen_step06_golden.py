"""本 6.4(本家 step06/neuralnet.py)を実走し、vol3 とのパリティ検証用 golden を生成する。

「PyTorch パリティ検証」の初リハーサル(学習過程のパリティ — VGG16 の forward パリティの次段)。
本家スクリプトに忠実(seed 0、x=rand(100,1)、y=sin(2πx)+一様ノイズ、1→10→1 sigmoid、
SGD lr=0.2、10000 iter、mse_loss)。ただし乱数生成そのものは移植しない —
データと初期重みを npz で輸出し、Rust 側はそれを読み込むことで RNG の差を無効化する。

出力: vol5/dataset/step06_golden.npz(gitignore 済み・本スクリプトで再生成可能)
  x, y            : 学習データ (100,1) f32
  l0/W, l0/b, l1/W, l1/b : 初期重み。vol3 MLP の named_params キーに合わせ、
                    W は PyTorch の (out,in) から vol3 の (in,out) へ転置済み。
                    → vol3 の load_weights がそのまま読める(余分なキーは無視される)
  loss_history    : 全 10000 iter の loss (f32)
  final/l0/W 等   : 学習後の重み
  x_grid, y_grid  : 学習後モデルの linspace(0,1,100) 上の予測(forward パリティ用)

実行: scratchpad の venv-torch などで `python gen_step06_golden.py`(要 torch, numpy)
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
x = torch.rand(100, 1)
y = torch.sin(2 * torch.pi * x) + torch.rand(100, 1)


class Model(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        y = self.linear1(x)
        y = F.sigmoid(y)
        y = self.linear2(y)
        return y


def state_np(model):
    # PyTorch の weight は (out, in) — vol3 Linear は (in, out) なので転置する
    sd = model.state_dict()
    return {
        "l0/W": sd["linear1.weight"].numpy().T.copy(),
        "l0/b": sd["linear1.bias"].numpy().copy(),
        "l1/W": sd["linear2.weight"].numpy().T.copy(),
        "l1/b": sd["linear2.bias"].numpy().copy(),
    }


model = Model()
init_params = state_np(model)

lr = 0.2
iters = 10000
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

loss_history = np.zeros(iters, dtype=np.float32)
for i in range(iters):
    y_pred = model(x)
    loss = F.mse_loss(y, y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history[i] = np.float32(loss.item())
    if i % 1000 == 0:
        print(f"iter {i}: loss = {loss.item():.6f}")
print(f"final: loss = {loss_history[-1]:.6f}")

final_params = {f"final/{k}": v for k, v in state_np(model).items()}
x_grid = torch.linspace(0, 1, 100).reshape(-1, 1)
with torch.no_grad():
    y_grid = model(x_grid).numpy()

out_path = Path(__file__).resolve().parent.parent / "dataset" / "step06_golden.npz"
out_path.parent.mkdir(exist_ok=True)
np.savez(
    out_path,
    x=x.numpy(),
    y=y.numpy(),
    loss_history=loss_history,
    x_grid=x_grid.numpy(),
    y_grid=y_grid,
    **init_params,
    **final_params,
)
print(f"saved: {out_path}")
for k, v in {**init_params, **final_params}.items():
    print(f"  {k}: shape {v.shape} dtype {v.dtype}")
