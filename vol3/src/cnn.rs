use ndarray::{Array4, Array6, ArrayD, Ix4, s};
use std::ops::AddAssign;

use crate::function::Function;
use crate::variable::Variable;

pub fn im2col_array(
    img: &ArrayD<f32>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
    to_matrix: bool,
) -> ArrayD<f32> {
    let img_4d = img
        .view()
        .into_dimensionality::<Ix4>()
        .expect("im2col requires 4D input");
    let (n, c, h, w) = img_4d.dim();
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = pad;

    let out_h = (h + 2 * ph - kh) / sh + 1;
    let out_w = (w + 2 * pw - kw) / sw + 1;

    let mut col = Array6::<f32>::zeros((n, c, kh, kw, out_h, out_w));

    let mut img_padded = Array4::<f32>::zeros((n, c, h + 2 * ph, w + 2 * pw));
    img_padded
        .slice_mut(s![.., .., ph..ph + h, pw..pw + w])
        .assign(&img_4d);

    for y in 0..kh {
        for x in 0..kw {
            let slice = img_padded.slice(s![.., .., y..;sh, x..;sw]);
            let slice = slice.slice(s![.., .., 0..out_h, 0..out_w]);
            col.slice_mut(s![.., .., y, x, .., ..]).assign(&slice);
        }
    }

    if to_matrix {
        let col = col.permuted_axes((0, 4, 5, 1, 2, 3));
        let col = col.as_standard_layout().into_owned();
        let flattened = col
            .into_shape_with_order((n * out_h * out_w, c * kh * kw))
            .unwrap();
        flattened.into_dyn()
    } else {
        col.into_dyn()
    }
}

pub fn col2im_array(
    col: &ArrayD<f32>,
    input_shape: (usize, usize, usize, usize),
    kernel_size: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
    to_matrix: bool,
) -> ArrayD<f32> {
    let (n, c, h, w) = input_shape;
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = pad;

    let out_h = (h + 2 * ph - kh) / sh + 1;
    let out_w = (w + 2 * pw - kw) / sw + 1;

    let col_6d = if to_matrix {
        let col_2d = col
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .expect("col2im requires 2D input when to_matrix=true");
        let col_reshaped = col_2d
            .into_shape_with_order((n, out_h, out_w, c, kh, kw))
            .unwrap();
        col_reshaped.permuted_axes((0, 3, 4, 5, 1, 2)).to_owned()
    } else {
        col.view()
            .into_dimensionality::<ndarray::Ix6>()
            .expect("col2im requires 6D input when to_matrix=false")
            .to_owned()
    };

    let mut img = Array4::<f32>::zeros((n, c, h + 2 * ph + sh - 1, w + 2 * pw + sw - 1));
    for y in 0..kh {
        for x in 0..kw {
            let col_slice = col_6d.slice(s![.., .., y, x, .., ..]);
            let mut img_slice = img.slice_mut(s![.., .., y..;sh, x..;sw]);
            let mut img_slice = img_slice.slice_mut(s![.., .., 0..out_h, 0..out_w]);
            img_slice.add_assign(&col_slice);
        }
    }

    img.slice(s![.., .., ph..h + ph, pw..w + pw])
        .to_owned()
        .into_dyn()
}

pub struct Im2Col {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub pad: (usize, usize),
    pub to_matrix: bool,
}

impl Im2Col {
    pub fn new(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        pad: (usize, usize),
        to_matrix: bool,
    ) -> Self {
        Self {
            kernel_size,
            stride,
            pad,
            to_matrix,
        }
    }
}

impl crate::function::Forward for Im2Col {
    fn forward(&self, xs: &[ndarray::ArrayD<f32>]) -> ndarray::ArrayD<f32> {
        let x = &xs[0];
        im2col_array(x, self.kernel_size, self.stride, self.pad, self.to_matrix)
    }
}

impl Function for Im2Col {
    fn backward(&self, xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        let x = &xs[0];
        let input_shape = {
            let s = x.shape();
            (s[0], s[1], s[2], s[3])
        };
        vec![
            Col2Im::new(
                input_shape,
                self.kernel_size,
                self.stride,
                self.pad,
                self.to_matrix,
            )
            .call(std::slice::from_ref(gy)),
        ]
    }
}

pub struct Col2Im {
    pub input_shape: (usize, usize, usize, usize),
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub pad: (usize, usize),
    pub to_matrix: bool,
}

impl Col2Im {
    pub fn new(
        input_shape: (usize, usize, usize, usize),
        kernel_size: (usize, usize),
        stride: (usize, usize),
        pad: (usize, usize),
        to_matrix: bool,
    ) -> Self {
        Self {
            input_shape,
            kernel_size,
            stride,
            pad,
            to_matrix,
        }
    }
}

impl crate::function::Forward for Col2Im {
    fn forward(&self, xs: &[ndarray::ArrayD<f32>]) -> ndarray::ArrayD<f32> {
        let col = &xs[0];
        col2im_array(
            col,
            self.input_shape,
            self.kernel_size,
            self.stride,
            self.pad,
            self.to_matrix,
        )
    }
}

impl Function for Col2Im {
    fn backward(&self, _xs: &[Variable], gy: &Variable) -> Vec<Variable> {
        vec![
            Im2Col::new(self.kernel_size, self.stride, self.pad, self.to_matrix)
                .call(std::slice::from_ref(gy)),
        ]
    }
}

pub fn conv2d_simple(
    x: &Variable,
    w: &Variable,
    b: Option<&Variable>,
    stride: (usize, usize),
    pad: (usize, usize),
) -> Variable {
    let (n, _c, h, _w_dim) = {
        let s = x.shape();
        (s[0], s[1], s[2], s[3])
    };
    let (oc, _c, kh, kw) = {
        let s = w.shape();
        (s[0], s[1], s[2], s[3])
    };

    let (sh, sw) = stride;
    let (ph, pw) = pad;

    let out_h = (h + 2 * ph - kh) / sh + 1;
    let out_w = (_w_dim + 2 * pw - kw) / sw + 1;

    let col = Im2Col::new((kh, kw), stride, pad, true).call(std::slice::from_ref(x));

    let w_flat = w.reshape(&[oc, _c * kh * kw]).transpose();

    let mut t = col.matmul(&w_flat);
    if let Some(bias) = b {
        t = &t + bias;
    }

    t.reshape(&[n, out_h, out_w, oc])
        .transpose_axes(&[0, 3, 1, 2])
}

pub fn pooling_simple(
    x: &Variable,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
) -> Variable {
    let s = x.shape();
    let (n, c, h, w) = (s[0], s[1], s[2], s[3]);
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = pad;
    let out_h = (h + 2 * ph - kh) / sh + 1;
    let out_w = (w + 2 * pw - kw) / sw + 1;

    let col = Im2Col::new(kernel_size, stride, pad, true).call(std::slice::from_ref(x));
    let col = col.reshape(&[n * c * out_h * out_w, kh * kw]);

    let y = col.max_axis(1, false);
    y.reshape(&[n, out_h, out_w, c])
        .transpose_axes(&[0, 3, 1, 2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    #[test]
    fn test_im2col_array() {
        // (1, 3, 7, 7) kernel=5, stride=1, pad=0 -> (9, 75)
        let x = Array4::<f32>::zeros((1, 3, 7, 7)).into_dyn();
        let col = im2col_array(&x, (5, 5), (1, 1), (0, 0), true);
        assert_eq!(col.dim(), ndarray::IxDyn(&[9, 75]));
    }

    #[test]
    fn test_col2im_array_duality_no_overlap() {
        let x = Array4::from_shape_vec((1, 1, 4, 4), (0..16).map(|x| x as f32).collect())
            .unwrap()
            .into_dyn();
        let col = im2col_array(&x, (2, 2), (2, 2), (0, 0), true);
        let x_rec = col2im_array(&col, (1, 1, 4, 4), (2, 2), (2, 2), (0, 0), true);
        assert_eq!(x, x_rec);
    }

    #[test]
    fn test_col2im_array_duality_with_overlap() {
        let x = Array4::from_shape_vec((1, 1, 4, 4), (0..16).map(|x| x as f32).collect())
            .unwrap()
            .into_dyn();
        let col = im2col_array(&x, (2, 2), (1, 1), (0, 0), true);
        let x_rec = col2im_array(&col, (1, 1, 4, 4), (2, 2), (1, 1), (0, 0), true);

        // With overlap, the reconstructed values will be summed.
        // corners = 1x, edges = 2x, interior = 4x
        assert_eq!(x_rec[[0, 0, 0, 0]], x[[0, 0, 0, 0]] * 1.0); // top-left
        assert_eq!(x_rec[[0, 0, 0, 1]], x[[0, 0, 0, 1]] * 2.0); // top edge
        assert_eq!(x_rec[[0, 0, 1, 1]], x[[0, 0, 1, 1]] * 4.0); // interior
    }
}
