use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use vol3::datasets::{Dataset, MnistDataset};
use vol3::functions::softmax_cross_entropy_simple;
use vol3::layers::{Layer, MLP};
use vol3::optimizers::{Optimizer, SGD};
use vol3::utils::approx_equal_arrayd;
use vol3::variable::Variable;

#[test]
fn test_relu_gradient_check() {
    // 0.0付近のテストを避け、確実に傾きが0か1になる点を選ぶ (x=0のキンク問題回避)
    let x_data = ndarray::array![[-1.5, -0.5, 0.5, 1.5], [2.0, -2.0, 0.1, -0.1]].into_dyn();
    let x = Variable::new(x_data.clone());

    // forward check
    let y = x.relu();
    let expected_y = ndarray::array![[0.0, 0.0, 0.5, 1.5], [2.0, 0.0, 0.1, 0.0]].into_dyn();
    assert!(approx_equal_arrayd(&y.data(), &expected_y, 1e-4));

    // backward check
    y.set_grad(Variable::new(
        ndarray::Array::ones(x_data.shape()).into_dyn(),
    ));
    y.backward(true, true); // retain_grad=true, create_graph=true

    let expected_gx = ndarray::array![[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0]].into_dyn();
    assert!(approx_equal_arrayd(&x.grad().unwrap(), &expected_gx, 1e-4));

    // double backprop check
    let gx = x.grad_var().unwrap();
    gx.set_grad(Variable::new(
        ndarray::Array::from_elem(x_data.shape(), 2.0).into_dyn(),
    ));
    gx.backward(false, false);

    // relu の二階微分 (x の勾配) は mask 定数のため 0 になるべきだが、
    // gy = y.grad_var() への勾配 (ggx -> gy) は mask 自体になる。
    // gy の勾配が mask * ggy となることを確認。
    let gy = y.grad_var().unwrap();
    let expected_ggy = ndarray::array![[0.0, 0.0, 2.0, 2.0], [2.0, 0.0, 2.0, 0.0]].into_dyn();
    assert!(approx_equal_arrayd(
        &gy.grad().unwrap(),
        &expected_ggy,
        1e-4
    ));
}

#[test]
fn test_mnist_subset_overfit() {
    let dataset = MnistDataset::new(true, Some(vol3::datasets::mnist_flatten_normalize));
    let subset_size = 100;

    // 100件のサブセットデータセットを手動で作るか、DataLoader で途中で break する
    // DataLoader のイテレータから 100件だけ抽出して使い回すのが一番簡単
    let batch_size = 100;

    // 確定的RNG
    let mut model_rng = ChaCha8Rng::seed_from_u64(42);
    let model = MLP::new(&[784, 100, 10], Variable::relu, &mut model_rng);
    let mut optimizer = SGD::new(0.1);
    optimizer.setup(&model);

    // 最初の100件を取得
    let mut batch_x_data = ndarray::Array::zeros(vec![batch_size, 784]);
    let mut batch_t = Vec::with_capacity(batch_size);

    for i in 0..subset_size {
        let (x, t) = dataset.get_item(i);
        batch_x_data.index_axis_mut(ndarray::Axis(0), i).assign(&x);
        batch_t.push(t);
    }
    let batch_x = Variable::new(batch_x_data.into_dyn());

    let mut initial_loss = 0.0;
    let mut final_loss = 0.0;

    for epoch in 0..100 {
        let y = model.forward(&batch_x);
        let loss = softmax_cross_entropy_simple(&y, &batch_t);

        if epoch == 0 {
            initial_loss = loss.item();
        }
        if epoch == 99 {
            final_loss = loss.item();
        }

        model.cleargrads();
        loss.backward(false, false);
        optimizer.update();
    }

    assert!(
        final_loss < initial_loss * 0.1, // Loss should drop significantly (to < 10% of initial)
        "Loss did not drop significantly: initial={}, final={}",
        initial_loss,
        final_loss
    );
    assert!(
        final_loss < 0.1,
        "Loss is still too high after overfitting: {}",
        final_loss
    );
}
