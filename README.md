# micrograd-rs

[![CI](https://github.com/Dafenxz0/micrograd-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Dafenxz0/micrograd-rs/actions/workflows/ci.yml)

A pure-Rust port of Andrej Karpathy's [`micrograd`](https://github.com/karpathy/micrograd) — a tiny scalar-valued reverse-mode autograd engine and minimal PyTorch-like neural network library with **zero external dependencies**.

## Features

- **Scalar autograd DAG (`src/engine.rs`):** Dynamic computation graph over `Value` supporting `+`, `-`, `*`, `/`, negation, `powf`, `relu`, and reverse-mode `backward()`.
- **Neural net layers (`src/nn.rs`):** `Module` trait, `Neuron`, `Layer`, and `MLP` with `parameters()` and `zero_grad()`.
- **Zero external dependencies (`src/rng.rs`):** Uses only Rust's standard library (`std`) with a built-in deterministic SplitMix64 PRNG.
- **Graphviz DOT tracing:** Built-in `Value::to_dot()` to visualize forward values and gradients.
- **Verified parity:** Numerical outputs and gradients are tested directly against Python `micrograd` / PyTorch reference values.

## Example usage

```rust
use micrograd_rs::Value;

fn main() {
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

    println!("{:.4}", g.data()); // 24.7041
    g.backward();
    println!("{:.4}", a.grad()); // 138.8338 (dg/da)
    println!("{:.4}", b.grad()); // 645.5773 (dg/db)
}
```

## Training a 2-layer MLP (`demo.ipynb` port)

Train a `2 -> 16 -> 16 -> 1` MLP (337 parameters) on a synthetic 2-moons binary classification dataset using SVM max-margin loss, L2 regularization, and decaying SGD, then render the learned decision boundary in ASCII:

```bash
cargo run --example moons_mlp
```

## Graphviz visualization (`trace_graph.ipynb` port)

Generate a Graphviz `.dot` representation of a 2D neuron's forward and backward pass:

```bash
cargo run --example trace_neuron | dot -Tsvg > gout.svg
```

## Running tests

```bash
cargo test
```

## Project structure

```text
├── src/
│   ├── lib.rs       # Public crate exports
│   ├── engine.rs    # Value, Op DAG, backward() topological pass, Graphviz DOT export
│   ├── nn.rs        # Module trait, Neuron, Layer, MLP
│   └── rng.rs       # Dependency-free SplitMix64 PRNG
├── examples/
│   ├── moons_mlp.rs # 2-moons binary classifier + ASCII decision boundary
│   └── trace_neuron.rs
└── tests/
    └── engine_tests.rs
```

## Design notes

- Python's `micrograd` stores closures for `_backward`; this port uses an explicit `Op` enum (`Leaf`, `Add`, `Mul`, `Pow`, `Relu`) inside `Rc<RefCell<ValueInner>>` to avoid per-node closure heap allocations and make Graphviz DOT generation straightforward.
- Because the computation graph is a DAG (parent edges only point to children), reference cycles cannot form and nodes are deallocated cleanly on `Drop`.

## License

MIT — see [LICENSE](LICENSE). Originally created by [Andrej Karpathy](https://github.com/karpathy/micrograd).
