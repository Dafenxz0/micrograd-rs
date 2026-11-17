//! Minimal PyTorch-like neural network layers built on top of [`Value`].

use std::fmt;

use crate::engine::Value;
use crate::rng::Rng;

/// Common interface for neural network modules with trainable parameters.
pub trait Module {
    /// Returns all trainable scalar parameters in this module.
    fn parameters(&self) -> Vec<Value>;

    /// Resets the gradient of every parameter to zero.
    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }
}

/// A single artificial neuron with `nin` inputs and an optional ReLU nonlinearity.
#[derive(Clone, Debug)]
pub struct Neuron {
    pub w: Vec<Value>,
    pub b: Value,
    pub nonlin: bool,
}

impl Neuron {
    /// Initializes weights uniformly in `[-1.0, 1.0)` and bias to `0.0`.
    pub fn new(nin: usize, nonlin: bool, rng: &mut Rng) -> Self {
        let w = (0..nin)
            .map(|_| Value::new(rng.gen_range(-1.0, 1.0)))
            .collect();
        Self {
            w,
            b: Value::new(0.0),
            nonlin,
        }
    }

    /// Computes `sum(w_i * x_i) + b`, followed by ReLU if `nonlin` is true.
    pub fn forward(&self, x: &[Value]) -> Value {
        assert_eq!(
            x.len(),
            self.w.len(),
            "expected {} inputs, got {}",
            self.w.len(),
            x.len()
        );
        let act = self
            .w
            .iter()
            .zip(x.iter())
            .fold(self.b.clone(), |acc, (wi, xi)| acc + wi * xi);
        if self.nonlin {
            act.relu()
        } else {
            act
        }
    }
}

impl Module for Neuron {
    fn parameters(&self) -> Vec<Value> {
        let mut params = self.w.clone();
        params.push(self.b.clone());
        params
    }
}

impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = if self.nonlin { "ReLU" } else { "Linear" };
        write!(f, "{kind}Neuron({})", self.w.len())
    }
}

/// A fully connected layer of [`Neuron`]s.
#[derive(Clone, Debug)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    /// Creates a layer mapping `nin` inputs to `nout` outputs.
    pub fn new(nin: usize, nout: usize, nonlin: bool, rng: &mut Rng) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(nin, nonlin, rng)).collect();
        Self { neurons }
    }

    /// Evaluates every neuron in the layer on input slice `x`.
    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(x)).collect()
    }
}

impl Module for Layer {
    fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(Module::parameters).collect()
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.neurons.iter().map(ToString::to_string).collect();
        write!(f, "Layer of [{}]", parts.join(", "))
    }
}

/// Multi-Layer Perceptron with ReLU hidden layers and a linear output layer.
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug)]
pub struct MLP {
    pub layers: Vec<Layer>,
}

impl MLP {
    /// Builds an MLP with input dimension `nin` and layer widths `nouts`.
    pub fn new(nin: usize, nouts: &[usize], rng: &mut Rng) -> Self {
        assert!(!nouts.is_empty(), "MLP must have at least one output layer");
        let mut sizes = Vec::with_capacity(nouts.len() + 1);
        sizes.push(nin);
        sizes.extend_from_slice(nouts);

        let last_idx = nouts.len() - 1;
        let layers = (0..nouts.len())
            .map(|i| Layer::new(sizes[i], sizes[i + 1], i != last_idx, rng))
            .collect();
        Self { layers }
    }

    /// Runs a forward pass through all layers.
    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        let mut current = x.to_vec();
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        current
    }

    /// Convenience forward pass for single-output MLPs.
    pub fn forward_scalar(&self, x: &[Value]) -> Value {
        let out = self.forward(x);
        assert_eq!(
            out.len(),
            1,
            "forward_scalar requires a single output neuron, got {}",
            out.len()
        );
        out.into_iter().next().expect("checked len == 1")
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(Module::parameters).collect()
    }
}

impl fmt::Display for MLP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.layers.iter().map(ToString::to_string).collect();
        write!(f, "MLP of [{}]", parts.join(", "))
    }
}
