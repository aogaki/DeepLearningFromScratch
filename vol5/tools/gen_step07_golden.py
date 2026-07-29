"""本 7.4(本家 step07/vae.py)のパリティ golden を生成する。

三段 tol ドクトリン(docs/framework-v1-notes.md)に基づく 2 種類の golden:

1) step07_golden.npz(既定・数秒):tier 1/2 用
   - MNIST 先頭 32 枚の固定バッチ(vol3/dataset の IDX から直接読む — Rust 側と同一ファイル)
   - 初期重み(vol5 の named_params キー・(in,out) 向きに転置済み)
   - eps_seq (10,32,20): 各 iter の reparameterize 乱数そのもの(輸出すれば RNG 差は消える)
   - loss_seq (10,): 固定バッチ+固定 eps で Adam(3e-4) を 10 iter 回した loss 軌跡
   - final10/...: 10 iter 後の重み(序盤の重み空間パリティ用)

2) --full 付き(数分): step07_epoch_curve.npz — tier 3 用
   本家どおり全 60k 画像・シャッフル・毎步 eps で 30 epoch 学習し、epoch 平均 loss 曲線を記録。
   Rust 側はこの曲線との「帯域一致」だけを主張する(長期の軌跡一致はカオスのため要求しない)。

実行: venv-torch などで `python gen_step07_golden.py [--full]`
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent  # リポジトリルート
OUT_DIR = Path(__file__).resolve().parent.parent / "dataset"

input_dim, hidden_dim, latent_dim = 784, 200, 20
learning_rate = 3e-4
batch_size = 32


def load_mnist_train_images():
    """vol3/dataset の IDX を直接読む(Rust 側と同一ソース)。(N,784) f32 [0,1]"""
    path = ROOT / "vol3" / "dataset" / "train-images-idx3-ubyte"
    raw = np.fromfile(path, dtype=np.uint8)
    n = int.from_bytes(raw[4:8].tobytes(), "big")
    images = raw[16:].reshape(n, 28 * 28)
    return (images.astype(np.float32) / 255.0), n


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.linear(x))
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, z):
        h = F.relu(self.linear1(z))
        return F.sigmoid(self.linear2(h))


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def get_loss(self, x, eps=None):
        mu, sigma = self.encoder(x)
        if eps is None:
            eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        x_hat = self.decoder(z)
        L1 = F.mse_loss(x_hat, x, reduction="sum")
        L2 = -torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)
        return (L1 + L2) / len(x)


# vol5 の named_params キー(enc/l1 等)への対応表。W は (out,in)→(in,out) に転置
KEYMAP = {
    "encoder.linear": "enc/l1",
    "encoder.linear_mu": "enc/l_mu",
    "encoder.linear_logvar": "enc/l_ln_var",
    "decoder.linear1": "dec/l1",
    "decoder.linear2": "dec/l2",
}


def state_np(model, prefix=""):
    sd = model.state_dict()
    out = {}
    for torch_name, vol5_name in KEYMAP.items():
        out[f"{prefix}{vol5_name}/W"] = sd[f"{torch_name}.weight"].numpy().T.copy()
        out[f"{prefix}{vol5_name}/b"] = sd[f"{torch_name}.bias"].numpy().copy()
    return out


def tier12():
    torch.manual_seed(0)
    model = VAE()
    init_params = state_np(model)

    images, _ = load_mnist_train_images()
    x = torch.from_numpy(images[:batch_size])  # 固定バッチ(シャッフルなし)

    iters = 10
    eps_seq = torch.randn(iters, batch_size, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_seq = np.zeros(iters, dtype=np.float32)
    for i in range(iters):
        optimizer.zero_grad()
        loss = model.get_loss(x, eps=eps_seq[i])
        loss.backward()
        optimizer.step()
        loss_seq[i] = np.float32(loss.item())
        print(f"iter {i}: loss = {loss.item():.6f}")

    final_params = state_np(model, prefix="final10/")
    out_path = OUT_DIR / "step07_golden.npz"
    np.savez(
        out_path,
        x_batch=x.numpy(),
        eps_seq=eps_seq.numpy(),
        loss_seq=loss_seq,
        **init_params,
        **final_params,
    )
    print(f"saved: {out_path}")


def tier3_full():
    torch.manual_seed(0)
    model = VAE()
    images, n = load_mnist_train_images()
    data = torch.from_numpy(images)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 30
    g = torch.Generator().manual_seed(1)
    epoch_losses = np.zeros(epochs, dtype=np.float32)
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=g)
        loss_sum, cnt = 0.0, 0
        for s in range(0, n - batch_size + 1, batch_size):
            x = data[perm[s : s + batch_size]]
            optimizer.zero_grad()
            loss = model.get_loss(x)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            cnt += 1
        epoch_losses[epoch] = np.float32(loss_sum / cnt)
        print(f"epoch {epoch + 1}: avg loss = {epoch_losses[epoch]:.4f}", flush=True)

    out_path = OUT_DIR / "step07_epoch_curve.npz"
    np.savez(out_path, epoch_losses=epoch_losses)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    if "--full" in sys.argv:
        tier3_full()
    else:
        tier12()
