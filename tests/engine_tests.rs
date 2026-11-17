use micrograd_rs::{Module, Rng, Value, MLP};

#[test]
fn test_sanity_check() {
    let x = Value::new(-4.0);
    let z = 2.0 * &x + 2.0 + &x;
    let q = z.relu() + &z * &x;
    let h = (&z * &z).relu();
    let y = &h + &q + &q * &x;
    y.backward();

    // Matches Python micrograd + PyTorch reference exactly
    assert_eq!(y.data(), -20.0);
    assert_eq!(x.grad(), 46.0);
}

#[test]
fn test_more_ops() {
    let a = Value::new(-4.0);
    let b = Value::new(2.0);
    let c = &a + &b;
    let d = &a * &b + b.powf(3.0);
    let c = &c + &c + 1.0;
    let c = &c + 1.0 + &c + (-&a);
    let d = &d + &d * 2.0 + (&b + &a).relu();
    let d = &d + 3.0 * &d + (&b - &a).relu();
    let e = c - d;
    let f = e.powf(2.0);
    let g = &f / 2.0;
    let g = g + 10.0 / &f;
    g.backward();

    let tol = 1e-6;
    assert!((g.data() - 24.70408163265306).abs() < tol);
    assert!((a.grad() - 138.83381924198252).abs() < tol);
    assert!((b.grad() - 645.5772594752186).abs() < tol);
}

#[test]
fn test_shared_node_gradients() {
    let x = Value::new(3.0);
    let y = &x + &x;
    y.backward();
    assert_eq!(x.grad(), 2.0);

    x.zero_grad();
    let z = &x * &x;
    z.backward();
    assert_eq!(x.grad(), 6.0);

    x.zero_grad();
    let w = &x / &x;
    w.backward();
    assert!((w.data() - 1.0).abs() < 1e-12);
    assert!(x.grad().abs() < 1e-12);
}

#[test]
fn test_mlp_training_reduces_loss() {
    let mut rng = Rng::new(42);
    let model = MLP::new(2, &[4, 1], &mut rng);
    assert_eq!(model.parameters().len(), 17);

    let dataset = [([1.0, 1.0], 1.0), ([-1.0, -1.0], -1.0)];
    let eval_loss = |m: &MLP| -> (Value, f64) {
        let mut losses = Vec::new();
        for (x, y) in dataset {
            let inputs = [Value::new(x[0]), Value::new(x[1])];
            let score = m.forward_scalar(&inputs);
            losses.push((1.0 + (-y) * score).relu());
        }
        let loss = losses.into_iter().sum::<Value>() * 0.5;
        let val = loss.data();
        (loss, val)
    };

    let (_, initial_loss) = eval_loss(&model);
    for _ in 0..30 {
        let (loss, _) = eval_loss(&model);
        model.zero_grad();
        loss.backward();
        for p in model.parameters() {
            p.set_data(p.data() - 0.1 * p.grad());
        }
    }
    let (_, final_loss) = eval_loss(&model);
    assert!(
        final_loss < initial_loss,
        "expected loss to drop: initial={initial_loss}, final={final_loss}"
    );
}
