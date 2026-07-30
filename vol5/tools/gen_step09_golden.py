"""本 9 章(本家 step09/diffusion_model.py)のパリティ golden を生成する。

三段 tol ドクトリンの tier 1/2 用。tier 3(縮小構成の学習曲線)は Rust 側の
1 バッチ実測を見てから構成を決める(このスクリプトには含まれない)。

内容(すべて f32):
  - UNet(in_ch=1, time_embed_dim=100)の初期重みを vol5/src/unet.rs の
    named_params キーへ写像して輸出(Conv は PyTorch と同レイアウトで転置不要、
    Linear のみ (out,in)→(in,out) 転置。BN は gamma/beta/running_mean/running_var)
  - 固定バッチ x(MNIST 先頭 16 枚、vol3/dataset の IDX と同一ソース)
  - 5 iter 分の t_seq(randint 1..=1000)と noise_seq(q_sample 用 ε)を輸出
    → 乱数差を無効化した Adam(lr=1e-3)5 iter の loss_seq
    【注意】loss = F.mse_loss(noise, noise_pred) は全要素平均(÷N·C·H·W)。
    vol3 の mean_squared_error(÷N)とは 784 倍違う — Rust 側で正規化を合わせること
  - final5/...: 5 iter 後の全重み(BN running stats 込み — 序盤の重み空間パリティ用)
  - サンプリング部分軌道: 学習後モデルを eval にして、固定 x_start (4,1,28,28) から
    t=10→1 の 10 ステップを固定 z_seq で denoise した x_after10(eval BN 経路+
    逆過程の式+スケジューラ配線の end-to-end 検証。t=1 の z はゼロで輸出)

実行: venv-torch などで `python gen_step09_golden.py`
出力: vol5/dataset/step09_golden.npz
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent.parent / "dataset"

num_timesteps = 1000
lr = 1e-3
batch_size = 16
iters = 5


def load_mnist_batch(n):
    path = ROOT / "vol3" / "dataset" / "train-images-idx3-ubyte"
    raw = np.fromfile(path, dtype=np.uint8)
    images = raw[16:].reshape(-1, 1, 28, 28)
    return torch.from_numpy((images[:n].astype(np.float32) / 255.0))


# ===== 本家 diffusion_model.py の忠実な写し(モデル+スケジューラ) =====

def _pos_encoding(time_idx, output_dim):
    t, D = time_idx, output_dim
    v = torch.zeros(D)
    i = torch.arange(0, D)
    div_term = torch.exp(i / D * math.log(10000))
    v[0::2] = torch.sin(t / div_term[0::2])
    v[1::2] = torch.cos(t / div_term[1::2])
    return v


def pos_encoding(timesteps, output_dim):
    batch_size = len(timesteps)
    v = torch.zeros(batch_size, output_dim)
    for i in range(batch_size):
        v[i] = _pos_encoding(timesteps[i], output_dim)
    return v


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch),
        )

    def forward(self, x, v):
        N, C, _, _ = x.shape
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)
        return self.convs(x + v)


class UNet(nn.Module):
    def __init__(self, in_ch=1, time_embed_dim=100):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_ch, 1)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x, timesteps):
        v = pos_encoding(timesteps, self.time_embed_dim)
        x1 = self.down1(x, v)
        x = self.maxpool(x1)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)
        x = self.bot1(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        return self.out(x)


class Diffuser:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise_with(self, x_0, t, noise):
        t_idx = t - 1
        alpha_bar = self.alpha_bars[t_idx].view(len(t), 1, 1, 1)
        return torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise

    def denoise_with(self, model, x, t_scalar, z):
        t = torch.tensor([t_scalar] * len(x), dtype=torch.long)
        t_idx = t - 1
        alpha = self.alphas[t_idx].view(len(x), 1, 1, 1)
        alpha_bar = self.alpha_bars[t_idx].view(len(x), 1, 1, 1)
        if t_scalar == 1:
            alpha_bar_prev = torch.ones_like(alpha_bar)
        else:
            alpha_bar_prev = self.alpha_bars[t_idx - 1].view(len(x), 1, 1, 1)
        with torch.no_grad():
            eps = model(x, t)
        mu = (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
        return mu + z * std


# ===== vol5/src/unet.rs の named_params キーへの写像 =====

BLOCKS = ["down1", "down2", "bot1", "up2", "up1"]
SUBMAP = {
    "convs.0": ("conv1", "conv"),
    "convs.1": ("bn1", "bn"),
    "convs.3": ("conv2", "conv"),
    "convs.4": ("bn2", "bn"),
    "mlp.0": ("mlp_fc1", "lin"),
    "mlp.2": ("mlp_fc2", "lin"),
}


def state_np(model, prefix=""):
    sd = model.state_dict()
    out = {}
    for blk in BLOCKS:
        for py_sub, (rs_sub, kind) in SUBMAP.items():
            base_py = f"{blk}.{py_sub}"
            base_rs = f"{prefix}{blk}/{rs_sub}"
            if kind == "conv":
                out[f"{base_rs}/W"] = sd[f"{base_py}.weight"].numpy().copy()  # 転置不要
                out[f"{base_rs}/b"] = sd[f"{base_py}.bias"].numpy().copy()
            elif kind == "lin":
                out[f"{base_rs}/W"] = sd[f"{base_py}.weight"].numpy().T.copy()  # (in,out) へ転置
                out[f"{base_rs}/b"] = sd[f"{base_py}.bias"].numpy().copy()
            elif kind == "bn":
                out[f"{base_rs}/gamma"] = sd[f"{base_py}.weight"].numpy().copy()
                out[f"{base_rs}/beta"] = sd[f"{base_py}.bias"].numpy().copy()
                out[f"{base_rs}/running_mean"] = sd[f"{base_py}.running_mean"].numpy().copy()
                out[f"{base_rs}/running_var"] = sd[f"{base_py}.running_var"].numpy().copy()
    out[f"{prefix}out/W"] = sd["out.weight"].numpy().copy()
    out[f"{prefix}out/b"] = sd["out.bias"].numpy().copy()
    return out


def tier3():
    """縮小構成(1024 枚・batch 64・5 epoch)の学習曲線 — tier 3 band 用。
    manual_seed(0) なので初期重みは tier1/2 golden と同一(Rust 側は golden の
    初期重みを load_weights してから、シャッフル・t・ノイズは自前の乱数で学習する
    = 統計的パリティ。曲線の band と突き合わせる)。"""
    torch.manual_seed(0)
    model = UNet()
    diffuser = Diffuser(num_timesteps)
    x_all = load_mnist_batch(1024)
    bsz, epochs = 64, 5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(7)
    model.train()
    epoch_losses = np.zeros(epochs, dtype=np.float32)
    for epoch in range(epochs):
        perm = torch.randperm(1024, generator=g)
        loss_sum, cnt = 0.0, 0
        for s in range(0, 1024, bsz):
            x = x_all[perm[s : s + bsz]]
            t = torch.randint(1, num_timesteps + 1, (bsz,), generator=g)
            noise = torch.randn(bsz, 1, 28, 28, generator=g)
            x_noisy = diffuser.add_noise_with(x, t, noise)
            noise_pred = model(x_noisy, t)
            loss = F.mse_loss(noise, noise_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            cnt += 1
        epoch_losses[epoch] = np.float32(loss_sum / cnt)
        print(f"epoch {epoch + 1}: avg loss = {epoch_losses[epoch]:.6f}", flush=True)
    out_path = OUT_DIR / "step09_tier3_curve.npz"
    np.savez(out_path, epoch_losses=epoch_losses)
    print(f"saved: {out_path}")


if "--tier3" in __import__("sys").argv:
    tier3()
    raise SystemExit

torch.manual_seed(0)
model = UNet()
diffuser = Diffuser(num_timesteps)
init_params = state_np(model)

x = load_mnist_batch(batch_size)
g = torch.Generator().manual_seed(1)
t_seq = torch.randint(1, num_timesteps + 1, (iters, batch_size), generator=g)
noise_seq = torch.randn(iters, batch_size, 1, 28, 28, generator=g)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
model.train()
loss_seq = np.zeros(iters, dtype=np.float32)
for i in range(iters):
    t = t_seq[i]
    noise = noise_seq[i]
    x_noisy = diffuser.add_noise_with(x, t, noise)
    noise_pred = model(x_noisy, t)
    loss = F.mse_loss(noise, noise_pred)  # 全要素平均(÷N·C·H·W)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_seq[i] = np.float32(loss.item())
    print(f"iter {i}: loss = {loss.item():.6f}")

final_params = state_np(model, prefix="final5/")

# ===== サンプリング部分軌道(eval モード、t=10→1、固定 z) =====
model.eval()
x_start = torch.randn(4, 1, 28, 28, generator=g)
z_seq = torch.randn(10, 4, 1, 28, 28, generator=g)
z_seq[9] = 0.0  # t=1 のステップはノイズなし(ゼロで輸出して両実装の意味論を無条件に一致させる)

x_cur = x_start.clone()
for k, t_scalar in enumerate(range(10, 0, -1)):
    x_cur = diffuser.denoise_with(model, x_cur, t_scalar, z_seq[k])
x_after10 = x_cur

out_path = OUT_DIR / "step09_golden.npz"
np.savez(
    out_path,
    x_batch=x.numpy(),
    t_seq=t_seq.numpy().astype(np.float32),
    noise_seq=noise_seq.numpy(),
    loss_seq=loss_seq,
    x_start=x_start.numpy(),
    z_seq=z_seq.numpy(),
    x_after10=x_after10.numpy(),
    **init_params,
    **final_params,
)
print(f"saved: {out_path}  (keys: {4 + 3 + 2 * (len(init_params))})")
