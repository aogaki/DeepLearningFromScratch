use ndarray::linalg::Dot;
use ndarray::{Array1, Array2, Array4, s};
use std::ops::AddAssign;

/// 本 7.4.2「im2colによる展開」(N,C,H,W) の入力からフィルタ適用領域を 1 行ずつ並べ、
/// (N·OH·OW, C·FH·FW) の 2 次元配列に展開する。畳み込みを行列積 1 回に変換する下準備
pub fn im2col(x: &Array4<f32>, fh: usize, fw: usize, stride: usize, pad: usize) -> Array2<f32> {
    let (n, c, h, w) = x.dim();
    let out_h = (h + 2 * pad - fh) / stride + 1;
    let out_w = (w + 2 * pad - fw) / stride + 1;

    let mut col = Array2::<f32>::zeros((n * out_h * out_w, c * fh * fw));

    let mut x_padded = Array4::<f32>::zeros((n, c, h + 2 * pad, w + 2 * pad));
    x_padded
        .slice_mut(s![.., .., pad..pad + h, pad..pad + w])
        .assign(x);

    for batch in 0..n {
        for channel in 0..c {
            for out_y in 0..out_h {
                for out_x in 0..out_w {
                    let y_start = out_y * stride;
                    let x_start = out_x * stride;
                    let patch = x_padded.slice(s![
                        batch,
                        channel,
                        y_start..y_start + fh,
                        x_start..x_start + fw
                    ]);
                    col.slice_mut(s![
                        batch * out_h * out_w + out_y * out_w + out_x,
                        channel * fh * fw..(channel + 1) * fh * fw
                    ])
                    .assign(&patch.iter().cloned().collect::<Array1<f32>>());
                }
            }
        }
    }
    col
}

/// 本 7.4(逆伝播用)im2col の逆写像。(N·OH·OW, C·FH·FW) の勾配を (N,C,H,W) に戻す。
/// 窓が重なる位置は加算(forward のファンアウト → backward で合算)、パディング部分は捨てる
pub fn col2im(
    col: &Array2<f32>,
    input_dim: (usize, usize, usize, usize),
    fh: usize,
    fw: usize,
    stride: usize,
    pad: usize,
) -> Array4<f32> {
    let (n, c, h, w) = input_dim;
    let out_h = (h + 2 * pad - fh) / stride + 1;
    let out_w = (w + 2 * pad - fw) / stride + 1;
    let mut img = Array4::<f32>::zeros((n, c, h + 2 * pad, w + 2 * pad));
    for batch in 0..n {
        for channel in 0..c {
            for out_y in 0..out_h {
                for out_x in 0..out_w {
                    let y_start = out_y * stride;
                    let x_start = out_x * stride;
                    let patch = col.slice(s![
                        batch * out_h * out_w + out_y * out_w + out_x,
                        channel * fh * fw..(channel + 1) * fh * fw
                    ]);
                    let patch_reshaped = patch.into_shape_with_order((fh, fw)).unwrap();
                    img.slice_mut(s![
                        batch,
                        channel,
                        y_start..y_start + fh,
                        x_start..x_start + fw
                    ])
                    .add_assign(&patch_reshaped);
                }
            }
        }
    }
    if pad > 0 {
        img.slice(s![.., .., pad..pad + h, pad..pad + w]).to_owned()
    } else {
        img
    }
}

/// 本 7.4.3「Convolutionレイヤの実装」フィルタ w (FN,C,FH,FW) とバイアス b (FN) を所有し、
/// forward は im2col → w との行列積 1 回 → (N,FN,OH,OW) へ整形で畳み込みを計算する
pub struct ConvolutionLayer {
    w: Array4<f32>,
    b: Array1<f32>,
    dw: Option<Array4<f32>>,
    db: Option<Array1<f32>>,
    stride: usize,
    pad: usize,
    x_shape: Option<(usize, usize, usize, usize)>,
    col: Option<Array2<f32>>,
}
impl ConvolutionLayer {
    pub fn new(w: Array4<f32>, b: Array1<f32>, stride: usize, pad: usize) -> Self {
        Self {
            w,
            b,
            dw: None,
            db: None,
            stride,
            pad,
            x_shape: None,
            col: None,
        }
    }

    pub fn forward(&mut self, x: &Array4<f32>) -> Array4<f32> {
        let (n, c, h, w) = x.dim();
        let (fn_, _fc, fh, fw) = self.w.dim();
        let out_h = (h + 2 * self.pad - fh) / self.stride + 1;
        let out_w = (w + 2 * self.pad - fw) / self.stride + 1;

        let col = im2col(x, fh, fw, self.stride, self.pad);
        let w_col = self
            .w
            .view()
            .into_shape_with_order((fn_, c * fh * fw))
            .unwrap();
        let mut out = col.dot(&w_col.t());
        out += &self.b;

        self.col = Some(col);
        self.x_shape = Some(x.dim());

        out.into_shape_with_order((n, out_h, out_w, fn_))
            .unwrap()
            .permuted_axes([0, 3, 1, 2])
    }

    pub fn backward(&mut self, dout: &Array4<f32>) -> Array4<f32> {
        let x_shape = self
            .x_shape
            .expect("forward must be called before backward");
        let col = self
            .col
            .as_ref()
            .expect("forward must be called before backward");

        let (_, _, out_h, out_w) = dout.dim();
        let (n, c, _, _) = x_shape;
        let (fn_, _, fh, fw) = self.w.dim();

        let dout_2d = dout
            .view()
            .permuted_axes([0, 2, 3, 1])
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order((n * out_h * out_w, fn_))
            .unwrap();

        self.db = Some(dout_2d.sum_axis(ndarray::Axis(0)));

        let dw_2d = dout_2d.t().dot(col);
        self.dw = Some(dw_2d.into_shape_with_order((fn_, c, fh, fw)).unwrap());

        let w_col = self
            .w
            .view()
            .into_shape_with_order((fn_, c * fh * fw))
            .unwrap();
        let dcol = dout_2d.dot(&w_col);
        col2im(&dcol, x_shape, fh, fw, self.stride, self.pad)
    }

    pub fn dw(&self) -> &Array4<f32> {
        self.dw.as_ref().expect("backward must be called before dw")
    }

    pub fn w(&self) -> &Array4<f32> {
        &self.w
    }
    pub fn w_mut(&mut self) -> &mut Array4<f32> {
        &mut self.w
    }

    pub fn w_and_dw(&mut self) -> (&mut Array4<f32>, &Array4<f32>) {
        (
            &mut self.w,
            self.dw.as_ref().expect("backward must be called before dw"),
        )
    }

    pub fn b(&self) -> &Array1<f32> {
        &self.b
    }
    pub fn b_mut(&mut self) -> &mut Array1<f32> {
        &mut self.b
    }

    pub fn db(&self) -> &Array1<f32> {
        self.db.as_ref().expect("backward must be called before db")
    }

    pub fn b_and_db(&mut self) -> (&mut Array1<f32>, &Array1<f32>) {
        (
            &mut self.b,
            self.db.as_ref().expect("backward must be called before db"),
        )
    }
}

/// 本 7.4.4「Poolingレイヤの実装」im2col で窓を展開し、1 行 = 1 チャンネル分の 1 窓に
/// reshape してから行ごとの max を取る。チャンネルごとに独立で、学習パラメータは持たない
pub struct PoolingLayer {
    pool_h: usize,
    pool_w: usize,
    stride: usize,
    pad: usize,
    x_shape: Option<(usize, usize, usize, usize)>,
    arg_max: Option<Array1<usize>>,
}
impl PoolingLayer {
    pub fn new(pool_h: usize, pool_w: usize, stride: usize, pad: usize) -> Self {
        Self {
            pool_h,
            pool_w,
            stride,
            pad,
            x_shape: None,
            arg_max: None,
        }
    }

    pub fn forward(&mut self, x: &Array4<f32>) -> Array4<f32> {
        self.x_shape = Some(x.dim());
        let (n, c, h, w) = x.dim();
        let out_h = (h + 2 * self.pad - self.pool_h) / self.stride + 1;
        let out_w = (w + 2 * self.pad - self.pool_w) / self.stride + 1;

        let col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad);
        let col_reshaped = col
            .into_shape_with_order((n * out_h * out_w * c, self.pool_h * self.pool_w))
            .unwrap();

        let mut out = Array1::<f32>::zeros(n * out_h * out_w * c);
        let mut argmax = Array1::<usize>::zeros(n * out_h * out_w * c);

        for (i, row) in col_reshaped.rows().into_iter().enumerate() {
            let (max_idx, max_val) =
                row.iter()
                    .enumerate()
                    .fold(
                        (0, f32::MIN),
                        |(idx, max), (i, &val)| {
                            if val > max { (i, val) } else { (idx, max) }
                        },
                    );
            argmax[i] = max_idx;
            out[i] = max_val;
        }
        self.arg_max = Some(argmax);

        out.into_shape_with_order((n, out_h, out_w, c))
            .unwrap()
            .permuted_axes([0, 3, 1, 2])
    }

    pub fn backward(&mut self, dout: &Array4<f32>) -> Array4<f32> {
        let pool_size = self.pool_h * self.pool_w;
        let argmax = self
            .arg_max
            .as_ref()
            .expect("forward must be called before backward");
        let x_shape = self
            .x_shape
            .expect("forward must be called before backward");
        let (n, c, _h, _w) = x_shape;
        let (_, _, out_h, out_w) = dout.dim();

        // dmax を経由せず、直接 dcol (N*OH*OW, C*pool_size) を作成する
        let mut dcol = Array2::<f32>::zeros((n * out_h * out_w, c * pool_size));

        let dout_permuted = dout.view().permuted_axes([0, 2, 3, 1]);

        for (i, (&idx, &val)) in argmax.iter().zip(dout_permuted.iter()).enumerate() {
            // i は C を最内ループとして進むため、パッチ行とチャンネルに分解できる
            let row = i / c;
            let channel = i % c;

            // 該当するパッチ行の、対象チャンネルのブロック (+idx) に値を散らす
            dcol[[row, channel * pool_size + idx]] = val;
        }

        col2im(
            &dcol,
            x_shape,
            self.pool_h,
            self.pool_w,
            self.stride,
            self.pad,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_im2col() {
        // shape テスト: 本 7.4.2 の例。(N,C,H,W)=(1,3,7,7)・5×5 → (N·OH·OW, C·FH·FW)=(9,75)
        let x = Array4::<f32>::zeros((1, 3, 7, 7));
        let col = im2col(&x, 5, 5, 1, 0);
        assert_eq!(col.dim(), (9, 75));

        // 値テスト: 4×4 連番・2×2 窓・stride2(重なりなし)。行順 (n,oy,ox)・行内 (py,px) の
        // レイアウトを全要素で検証。im2col は値を移動するだけで演算しないので exact 比較が正しい
        let x = Array4::from_shape_vec((1, 1, 4, 4), (0..16).map(|x| x as f32).collect()).unwrap();
        let col = im2col(&x, 2, 2, 2, 0);
        assert_eq!(col.dim(), (4, 4));
        let expect = Array2::from_shape_vec(
            (4, 4),
            vec![
                0., 1., 4., 5., 2., 3., 6., 7., 8., 9., 12., 13., 10., 11., 14., 15.,
            ],
        )
        .unwrap();
        assert_eq!(col, expect);

        // pad=1: ゼロパディング経路の検証。境界窓の外周に 0 が並ぶ位置まで含めて全要素照合
        let col_with_pad = im2col(&x, 2, 2, 2, 1);
        let expect_with_pad = Array2::from_shape_vec(
            (9, 4),
            vec![
                0., 0., 0., 0., 0., 0., 1., 2., 0., 0., 3., 0., 0., 4., 0., 8., 5., 6., 9., 10.,
                7., 0., 11., 0., 0., 12., 0., 0., 13., 14., 0., 0., 15., 0., 0., 0.,
            ],
        )
        .unwrap();
        assert_eq!(col_with_pad.dim(), (9, 4));
        assert_eq!(col_with_pad, expect_with_pad);
    }

    #[test]
    fn test_convolution_forward() {
        // shape テスト: 本 7.4 の例に近い構成。FN=10 本のフィルタで C:3→10 に変わることを確認
        let w =
            Array4::from_shape_vec((10, 3, 7, 7), (0..1470).map(|x| x as f32).collect()).unwrap();
        let b = Array1::from_shape_vec(10, (0..10).map(|x| x as f32).collect()).unwrap();
        let mut conv_layer = ConvolutionLayer::new(w, b, 1, 0);
        let x = Array4::from_shape_vec((1, 3, 7, 7), (0..147).map(|x| x as f32).collect()).unwrap();
        let out = conv_layer.forward(&x);
        assert_eq!(out.dim(), (1, 10, 1, 1));

        // 値テスト(全1フィルタ): 出力 = 各窓の総和になるので期待値が暗算できる
        let w_ones = Array4::from_shape_vec((1, 1, 2, 2), vec![1.0; 4]).unwrap();
        let b_zeros = Array1::from_shape_vec(1, vec![0.0]).unwrap();
        let mut conv_layer_ones = ConvolutionLayer::new(w_ones, b_zeros, 2, 0);
        let x_seq =
            Array4::from_shape_vec((1, 1, 4, 4), (0..16).map(|x| x as f32).collect()).unwrap();
        let out_seq = conv_layer_ones.forward(&x_seq);
        assert_eq!(out_seq.dim(), (1, 1, 2, 2));
        let expect_seq =
            Array4::from_shape_vec((1, 1, 2, 2), vec![10.0, 18.0, 42.0, 50.0]).unwrap();
        assert_eq!(out_seq, expect_seq);

        // 値テスト(デルタフィルタ): 1 箇所だけ 1 のフィルタなら出力 = その位置の入力値そのもの。
        // forward 内の w reshape (FN, C·FH·FW) と im2col の列順 (C,FH,FW) の整合をピンポイントで
        // 検証する(全1フィルタでは総和に混ざって列順のズレを検出できない)
        let mut w_delta = Array4::<f32>::zeros((1, 2, 2, 2));
        w_delta[[0, 1, 0, 0]] = 1.0; // チャンネル 1 の窓内位置 (0,0) にデルタを立てる
        let b_zero = Array1::<f32>::zeros(1);
        let mut conv_layer_delta = ConvolutionLayer::new(w_delta, b_zero, 2, 0);
        let x_seq =
            Array4::from_shape_vec((1, 2, 4, 4), (0..32).map(|x| x as f32).collect()).unwrap();
        let out_delta = conv_layer_delta.forward(&x_seq);
        assert_eq!(out_delta.dim(), (1, 1, 2, 2));
        // 期待値 = チャンネル 1 (値 16..32) の各窓左上
        let expect_delta =
            Array4::from_shape_vec((1, 1, 2, 2), vec![16.0, 18.0, 24.0, 26.0]).unwrap();
        assert_eq!(out_delta, expect_delta);
    }

    #[test]
    fn test_convolution_backward() {
        use ndarray::array;

        let mut w = Array4::<f32>::zeros((1, 2, 2, 2));
        w[[0, 1, 0, 0]] = 1.0;
        let b = Array1::<f32>::zeros(1);
        let mut conv = ConvolutionLayer::new(w, b, 2, 0);
        let mut x = Array4::<f32>::zeros((1, 2, 4, 4));
        for (i, v) in x.iter_mut().enumerate() {
            *v = (i % 16 + 100 * (i / 16)) as f32;
        }
        let _out = conv.forward(&x);

        // db
        let dout_seq = array![[[[10.0, 20.0], [30.0, 40.0]]]];
        let dx1 = conv.backward(&dout_seq);
        assert_eq!(conv.db()[[0]], 100.0);

        // dx
        let expected_dx = array![[
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [10.0, 0.0, 20.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [30.0, 0.0, 40.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ]];
        assert_eq!(dx1, expected_dx);

        // dw
        let dout_ones = Array4::<f32>::ones((1, 1, 2, 2));
        let _dx2 = conv.backward(&dout_ones);
        let expected_dw = array![[
            [[20.0, 24.0], [36.0, 40.0]],
            [[420.0, 424.0], [436.0, 440.0]]
        ]];
        assert_eq!(conv.dw(), expected_dw);

        // FN = 2
        let mut w = Array4::<f32>::zeros((2, 2, 2, 2));
        w[[0, 0, 0, 0]] = 1.0;
        w[[1, 1, 1, 1]] = 1.0;
        let b = Array1::<f32>::zeros(2);
        let mut conv = ConvolutionLayer::new(w, b, 2, 0);
        let mut x = Array4::<f32>::zeros((1, 2, 4, 4));
        for (i, v) in x.iter_mut().enumerate() {
            *v = (i % 16 + 100 * (i / 16)) as f32;
        }
        let _out = conv.forward(&x);
        let dout = array![[
            // FN = 0
            [[1.0, 2.0], [3.0, 4.0]],
            // FN = 1
            [[5.0, 6.0], [7.0, 8.0]]
        ]];
        let dx = conv.backward(&dout);

        // dx
        let expected_dx = array![[
            [
                [1.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 6.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 7.0, 0.0, 8.0],
            ]
        ]];
        assert_eq!(dx, expected_dx);

        // db
        let expected_db = array![10.0, 26.0];
        assert_eq!(conv.db(), expected_db);
    }

    #[test]
    fn test_pooling_forward() {
        // 値テスト: C=2 必須 — max 後の並び (n,oy,ox,c) を (N,C,OH,OW) へ戻す軸順バグは
        // C=1 だと偶然一致して検出できない。性質チェック: max の出力は必ず入力に存在する値
        let mut pooling_layer = PoolingLayer::new(2, 2, 2, 0);
        let x = Array4::from_shape_vec((1, 2, 4, 4), (0..32).map(|x| x as f32).collect()).unwrap();
        let out = pooling_layer.forward(&x);
        assert_eq!(out.dim(), (1, 2, 2, 2));
        let expect = Array4::from_shape_vec(
            (1, 2, 2, 2),
            vec![5.0, 7.0, 13.0, 15.0, 21.0, 23.0, 29.0, 31.0],
        )
        .unwrap();
        assert_eq!(out, expect);

        // N=2・C=2・4×6(非正方): バッチ/チャンネル/縦横の取り違え検出。N=1 では batch 項が
        // 消えて行インデックスのバグが見えず、正方形では out_h/out_w の転置が見えない。
        // 各 (batch,channel) ブロックが 24 ずつズレた連番なので、取り違えは値の差として現れる
        let mut pooling_layer = PoolingLayer::new(2, 2, 2, 0);
        let x = Array4::from_shape_vec((2, 2, 4, 6), (0..96).map(|x| x as f32).collect()).unwrap();
        let out = pooling_layer.forward(&x);
        assert_eq!(out.dim(), (2, 2, 2, 3));

        let expect = Array4::from_shape_vec(
            (2, 2, 2, 3),
            vec![
                7.0, 9.0, 11.0, 19.0, 21.0, 23.0, 31.0, 33.0, 35.0, 43.0, 45.0, 47.0, 55.0, 57.0,
                59.0, 67.0, 69.0, 71.0, 79.0, 81.0, 83.0, 91.0, 93.0, 95.0,
            ],
        )
        .unwrap();
        assert_eq!(out, expect);
    }

    #[test]
    fn test_pooling_backward() {
        use ndarray::array;

        // C == 2
        let mut pool = PoolingLayer::new(2, 2, 2, 0);

        let x = array![[
            [
                [1.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 4.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0, 0.0],
                [0.0, 3.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ]];

        let _out = pool.forward(&x);
        let dout = array![[[[10.0, 20.0], [30.0, 40.0],], [[50.0, 60.0], [70.0, 80.0],]]];

        let dx = pool.backward(&dout);
        let expected_dx = array![[
            [
                [10.0, 0.0, 0.0, 20.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [30.0, 0.0, 0.0, 40.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 50.0, 60.0, 0.0],
                [0.0, 70.0, 80.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ]];

        assert_eq!(dx, expected_dx);

        // n == 2
        let mut pool = PoolingLayer::new(2, 2, 2, 0);
        let x = array![
            [[[1.0, 0.0], [0.0, 0.0]]], // Batch 0: 左上が最大
            [[[0.0, 0.0], [0.0, 2.0]]]  // Batch 1: 右下が最大
        ];
        let _out = pool.forward(&x);
        let dout = array![[[[10.0]]], [[[20.0]]]];
        let dx = pool.backward(&dout);
        let expected_dx = array![
            [[[10.0, 0.0], [0.0, 0.0]]], // Batch 0 は左上に散る
            [[[0.0, 0.0], [0.0, 20.0]]]  // Batch 1 は右下に散る
        ];
        assert_eq!(dx, expected_dx);

        // pverlap test: 重なりあり (stride < 窓幅) の性質テスト
        let mut pool = PoolingLayer::new(2, 2, 1, 0);
        let mut x = Array4::<f32>::zeros((1, 1, 3, 3));
        x[[0, 0, 1, 1]] = 100.0;
        let _out = pool.forward(&x);
        let dout = Array4::<f32>::ones((1, 1, 2, 2));
        let dx = pool.backward(&dout);
        assert_eq!(dx[[0, 0, 1, 1]], 4.0);
        assert_eq!(dx.sum(), dout.sum());

        // pad test
        let mut pool = PoolingLayer::new(2, 2, 2, 1);
        let x = array![[[[1.0, 2.0], [3.0, 4.0]]]];
        let _out = pool.forward(&x);
        let dout = array![[[[10.0, 20.0], [30.0, 40.0]]]];
        let dx = pool.backward(&dout);
        assert_eq!(dx, dout);
    }

    #[test]
    fn test_col2im() {
        // 往復テスト 1(重なりなし: stride = 窓幅): col2im(im2col(x)) == x で完全往復
        let x = Array4::from_shape_vec((1, 1, 4, 4), (0..16).map(|x| x as f32).collect()).unwrap();
        let col = im2col(&x, 2, 2, 2, 0);
        let x_reconstructed = col2im(&col, (1, 1, 4, 4), 2, 2, 2, 0);
        assert_eq!(x, x_reconstructed);

        // 往復テスト 2(重なりあり: stride 1): col2im は重なりを加算するので、往復後の各
        // ピクセルは「自分をカバーする窓の数」倍になる(角=1, 辺=2, 内側=4)。
        // assign で上書きするバグ(勾配の消失)はここで落ちる
        let col = im2col(&x, 2, 2, 1, 0);
        let x_reconstructed = col2im(&col, (1, 1, 4, 4), 2, 2, 1, 0);
        let expected = Array4::from_shape_vec(
            (1, 1, 4, 4),
            vec![
                0.0, 2.0, 4.0, 3.0, 8.0, 20.0, 24.0, 14.0, 16.0, 36.0, 40.0, 22.0, 12.0, 26.0,
                28.0, 15.0,
            ],
        )
        .unwrap();
        assert_eq!(x_reconstructed, expected);

        // 往復テスト 3(pad=1・stride 1): col2im の pad>0 分岐(中央切り出し)を通す。
        // 元の 4×4 はパディング済み 6×6 の内側に収まるので全ピクセルが窓 4 個にカバーされ、
        // 期待値は一律 4·x になる
        let col = im2col(&x, 2, 2, 1, 1);
        let x_reconstructed = col2im(&col, (1, 1, 4, 4), 2, 2, 1, 1);
        let expected = Array4::from_shape_vec(
            (1, 1, 4, 4),
            vec![
                0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0,
                56.0, 60.0,
            ],
        )
        .unwrap();
        assert_eq!(x_reconstructed, expected);
    }
}
