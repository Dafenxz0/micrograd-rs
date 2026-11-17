//! Port of Karpathy's `demo.ipynb`: trains a 2-layer MLP on a 2-moons dataset
//! using SVM max-margin loss + L2 regularization, then draws the decision boundary in ASCII.

use std::f64::consts::PI;

use micrograd_rs::{Module, Rng, Value, MLP};

fn make_moons(n_samples: usize, noise: f64, rng: &mut Rng) -> Vec<([f64; 2], f64)> {
    let n_out = n_samples / 2;
    let n_in = n_samples - n_out;
    let mut data = Vec::with_capacity(n_samples);

    for i in 0..n_out {
        let angle = PI * (i as f64) / (n_out.max(1) as f64);
        let x = angle.cos() + noise * rng.gen_normal();
        let y = angle.sin() + noise * rng.gen_normal();
        data.push(([x, y], -1.0));
    }
    for i in 0..n_in {
        let angle = PI * (i as f64) / (n_in.max(1) as f64);
        let x = 1.0 - angle.cos() + noise * rng.gen_normal();
        let y = 1.0 - angle.sin() - 0.5 + noise * rng.gen_normal();
        data.push(([x, y], 1.0));
    }
    data
}

fn evaluate_loss(model: &MLP, dataset: &[([f64; 2], f64)]) -> (Value, f64) {
    let mut losses = Vec::with_capacity(dataset.len());
    let mut correct = 0usize;

    for (xy, y) in dataset {
        let inputs = [Value::new(xy[0]), Value::new(xy[1])];
        let score = model.forward_scalar(&inputs);
        if (score.data() > 0.0) == (*y > 0.0) {
            correct += 1;
        }
        losses.push((1.0 + (-y) * score).relu());
    }

    let data_loss = losses.into_iter().sum::<Value>() * (1.0 / dataset.len() as f64);
    let alpha = 1e-4;
    let reg_loss = alpha * model.parameters().iter().map(|p| p * p).sum::<Value>();
    let total_loss = data_loss + reg_loss;
    let accuracy = (correct as f64) / (dataset.len() as f64);
    (total_loss, accuracy)
}

fn print_decision_boundary(model: &MLP, dataset: &[([f64; 2], f64)]) {
    let x_min = -1.6;
    let x_max = 2.6;
    let y_min = -1.1;
    let y_max = 1.6;
    let cols = 64usize;
    let rows = 24usize;

    println!("\nDecision boundary (+: class +1, -: class -1, ' ' / '.' background):");
    for r in 0..rows {
        let y = y_max - (r as f64) / ((rows - 1) as f64) * (y_max - y_min);
        let mut line = String::with_capacity(cols);
        for c in 0..cols {
            let x = x_min + (c as f64) / ((cols - 1) as f64) * (x_max - x_min);

            let point_char = dataset.iter().find_map(|(pxy, label)| {
                let dx = (pxy[0] - x).abs();
                let dy = (pxy[1] - y).abs();
                if dx < (x_max - x_min) / (cols as f64 * 1.4)
                    && dy < (y_max - y_min) / (rows as f64 * 1.4)
                {
                    Some(if *label > 0.0 { '+' } else { '-' })
                } else {
                    None
                }
            });

            if let Some(ch) = point_char {
                line.push(ch);
            } else {
                let score = model.forward_scalar(&[Value::new(x), Value::new(y)]).data();
                line.push(if score > 0.0 { ':' } else { ' ' });
            }
        }
        println!("|{line}|");
    }
}

fn main() {
    let mut rng = Rng::new(1337);
    let dataset = make_moons(100, 0.1, &mut rng);

    let model = MLP::new(2, &[16, 16, 1], &mut rng);
    println!("{model}");
    println!("number of parameters: {}", model.parameters().len());

    for step in 0..100 {
        let (total_loss, acc) = evaluate_loss(&model, &dataset);

        model.zero_grad();
        total_loss.backward();

        let lr = 1.0 - 0.9 * (step as f64) / 100.0;
        for p in model.parameters() {
            p.set_data(p.data() - lr * p.grad());
        }

        if step % 10 == 0 || step == 99 {
            println!(
                "step {:>2} | loss {:.4} | accuracy {:>5.1}%",
                step,
                total_loss.data(),
                acc * 100.0
            );
        }
    }

    print_decision_boundary(&model, &dataset);
}
