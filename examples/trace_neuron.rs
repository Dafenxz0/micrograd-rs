//! Port of `trace_graph.ipynb`: builds a 2D neuron forward/backward graph and
//! prints Graphviz DOT output (`cargo run --example trace_neuron | dot -Tsvg > gout.svg`).

use micrograd_rs::{Neuron, Rng, Value};

fn main() {
    let mut rng = Rng::new(1337);
    let neuron = Neuron::new(2, true, &mut rng);
    let x = [Value::new(1.0), Value::new(-2.0)];
    let y = neuron.forward(&x);
    y.backward();

    println!("{}", y.to_dot());
}
