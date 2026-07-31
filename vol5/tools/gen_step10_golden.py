"""本 10 章(本家 step10/classifier_free_guidance.py)のパリティ golden を生成する。

gen_step09_golden.py との差分は本家どおり 3 点だけ:
  1. UNetCond = UNet + nn.Embedding(10, time_embed_dim)。forward で
     `if labels is not None: v += label_emb(labels)`(時刻埋め込みへの加算)
  2. 学習: バッチ丸ごとラベルドロップ(本家は np.random.random() < 0.1)。
     golden では乱数でなく drop_flags = [1,1,0,1,1] を手で固定し、
     5 iter の中で条件付き/無条件の両分岐を必ず踏ませる(iter 2 が None)
  3. サンプリング: CFG — 各ステップで条件付き/無条件の 2 回 forward、
     eps = eps_uncond + gamma * (eps_cond - eps_uncond)、gamma = 3.0

輸出(すべて f32、ラベルのみ i64):
  - 初期重み 83 本(step09 の 82 本 + label_emb/W)。label_emb/W は (10, 100) を
    転置なしで輸出 — vol3 側は one-hot @ W 合成なので PyTorch の
    nn.Embedding.weight とレイアウトがそのまま一致する
  - x_batch (16,1,28,28) / labels_batch (16,) i64 — MNIST 先頭 16 枚とその実ラベル
  - drop_flags (5,) u8(1=Some(labels), 0=None)/ t_seq / noise_seq / loss_seq
    【注意】loss = F.mse_loss は全要素平均 — Rust 側は ÷784 の正規化合わせが必要
  - final5/...: 5 iter 後の全 83 本(BN running stats 込み)
  - CFG サンプリング部分軌道: eval モード、labels_sample = [0,1,2,3]、gamma=3.0、
    固定 x_start (4,1,28,28) から t=10→1 を固定 z_seq で denoise した x_after10
    (t=1 の z はゼロで輸出)

実行: venv-torch などで `python gen_step10_golden.py`
出力: vol5/dataset/step10_golden.npz
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
gamma = 3.0
drop_flags = np.array([1, 1, 0, 1, 1], dtype=np.uint8)  # iter 2 だけ無条件


def load_mnist_batch(n):
    path = ROOT / "vol3" / "dataset" / "train-images-idx3-ubyte"
    raw = np.fromfile(path, dtype=np.uint8)
    images = raw[16:].reshape(-1, 1, 28, 28)
    return torch.from_numpy(images[:n].astype(np.float32) / 255.0)


def load_mnist_labels(n):
    path = ROOT / "vol3" / "dataset" / "train-labels-idx1-ubyte"
    raw = np.fromfile(path, dtype=np.uint8)
    return torch.from_numpy(raw[8 : 8 + n].astype(np.int64))


# ===== 本家 conditional.py / classifier_free_guidance.py の忠実な写し =====

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


class UNetCond(nn.Module):
    def __init__(self, in_ch=1, time_embed_dim=100, num_labels=None):
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
        if num_labels is not None:
            self.label_emb = nn.Embedding(num_labels, time_embed_dim)

    def forward(self, x, timesteps, labels=None):
        v = pos_encoding(timesteps, self.time_embed_dim)
        if labels is not None:
            v += self.label_emb(labels)
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

    def denoise_cfg_with(self, model, x, t_scalar, z, labels, gamma):
        """本家 classifier_free_guidance.py の denoise の写し(z 注入版)。"""
        t = torch.tensor([t_scalar] * len(x), dtype=torch.long)
        t_idx = t - 1
        alpha = self.alphas[t_idx].view(len(x), 1, 1, 1)
        alpha_bar = self.alpha_bars[t_idx].view(len(x), 1, 1, 1)
        if t_scalar == 1:
            alpha_bar_prev = torch.ones_like(alpha_bar)
        else:
            alpha_bar_prev = self.alpha_bars[t_idx - 1].view(len(x), 1, 1, 1)
        with torch.no_grad():
            eps = model(x, t, labels)
            eps_uncond = model(x, t)
            eps = eps_uncond + gamma * (eps - eps_uncond)
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
    # nn.Embedding の weight (10, 100) は one-hot @ W とレイアウト一致 — 転置不要
    out[f"{prefix}label_emb/W"] = sd["label_emb.weight"].numpy().copy()
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


torch.manual_seed(0)
model = UNetCond(num_labels=10)
diffuser = Diffuser(num_timesteps)
init_params = state_np(model)

x = load_mnist_batch(batch_size)
labels_batch = load_mnist_labels(batch_size)
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
    labels = labels_batch if drop_flags[i] == 1 else None  # バッチ丸ごとドロップ
    noise_pred = model(x_noisy, t, labels)
    loss = F.mse_loss(noise, noise_pred)  # 全要素平均(÷N·C·H·W)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_seq[i] = np.float32(loss.item())
    tag = "cond" if drop_flags[i] == 1 else "uncond"
    print(f"iter {i} ({tag}): loss = {loss.item():.6f}")

final_params = state_np(model, prefix="final5/")

# ===== CFG サンプリング部分軌道(eval モード、t=10→1、固定 z、gamma=3.0) =====
model.eval()
labels_sample = torch.tensor([0, 1, 2, 3], dtype=torch.long)
x_start = torch.randn(4, 1, 28, 28, generator=g)
z_seq = torch.randn(10, 4, 1, 28, 28, generator=g)
z_seq[9] = 0.0  # t=1 のステップはノイズなし

x_cur = x_start.clone()
for k, t_scalar in enumerate(range(10, 0, -1)):
    x_cur = diffuser.denoise_cfg_with(model, x_cur, t_scalar, z_seq[k], labels_sample, gamma)
x_after10 = x_cur

out_path = OUT_DIR / "step10_golden.npz"
np.savez(
    out_path,
    x_batch=x.numpy(),
    labels_batch=labels_batch.numpy(),
    drop_flags=drop_flags,
    t_seq=t_seq.numpy().astype(np.float32),
    noise_seq=noise_seq.numpy(),
    loss_seq=loss_seq,
    x_start=x_start.numpy(),
    z_seq=z_seq.numpy(),
    labels_sample=labels_sample.numpy(),
    x_after10=x_after10.numpy(),
    **init_params,
    **final_params,
)
print(f"saved: {out_path}")
print(f"init params: {len(init_params)} keys, final5: {len(final_params)} keys")
