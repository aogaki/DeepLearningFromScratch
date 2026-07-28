"""CartPole golden-trajectory generator(一度きりの下準備)。

vol4/src/cart_pole.rs のパリティテスト用 golden 値を、本家 gymnasium の
CartPoleEnv(classic_control/cartpole.py, float64)を実際に走らせて生成する。
Rust 側の転記ミスは自己一致テストでは検出できない(vol4 ch4 の教訓)ため、
独立実装との突き合わせを正とする。

実行方法(生成時は gymnasium 1.3.0 / Python 3.14):
    python3 -m venv /tmp/gymenv && /tmp/gymenv/bin/pip install gymnasium
    /tmp/gymenv/bin/python vol4/tools/gen_cartpole_golden.py
出力(Rust の配列リテラル)を cart_pole.rs のテストの GOLDEN_* 定数へ貼り付ける。
"""

import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv


def run(init, actions, stop_on_terminated=True):
    env = CartPoleEnv()
    env.reset(seed=0)  # np_random 等の初期化のため(状態は直後に上書き)
    env.state = np.array(init, dtype=np.float64)
    rows = []
    for a in actions:
        _obs, reward, terminated, _truncated, _info = env.step(a)
        state = np.asarray(env.state, dtype=np.float64).copy()
        rows.append((state, reward, terminated))
        if terminated and stop_on_terminated:
            break
    return rows


def emit(name, init, actions, rows):
    # リテラルは np.float32 の最短 round-trip 表現で出力する。float64 の桁を
    # そのまま貼ると f32 に載らない精度で clippy::excessive_precision になる。
    # f64→f32 の丸めは numpy でも Rust コンパイラでも同じ(最近接偶数)なので値は不変。
    print(f"    // {name}: init={init}, actions={actions[: len(rows)]}")
    print(f"    const {name}: [([f32; 4], f32, bool); {len(rows)}] = [")
    for i, (s, r, t) in enumerate(rows):
        vals = ", ".join(str(np.float32(v)) for v in s)
        flag = "true" if t else "false"
        print(f"        ([{vals}], {r:.1f}, {flag}), // step {i + 1}")
    print("    ];")
    print()


# 軌道1: 物理パリティ(非対称な初期状態+左右混合の行動列、終了しない範囲)
INIT1 = [0.01, -0.02, 0.03, -0.04]
ACTS1 = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
emit("GOLDEN_PHYSICS", INIT1, ACTS1, run(INIT1, ACTS1))

# 軌道2: 角度による終了境界(±12° = 0.20943951 rad を跨ぐステップの一致を確認)
INIT2 = [0.0, 0.0, 0.15, 1.0]
ACTS2 = [1] * 20
emit("GOLDEN_ANGLE_TERMINATION", INIT2, ACTS2, run(INIT2, ACTS2))
