//! Scalar-valued reverse-mode automatic differentiation engine.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashSet};
use std::fmt;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

/// Operation that produced a [`Value`] node in the computation DAG.
#[derive(Clone, Debug)]
pub enum Op {
    Leaf,
    Add(Value, Value),
    Mul(Value, Value),
    Pow(Value, f64),
    Relu(Value),
}

#[derive(Debug)]
struct ValueInner {
    data: f64,
    grad: f64,
    op: Op,
}

/// A scalar value participating in a dynamically built autograd DAG.
#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueInner>>);

impl Value {
    /// Creates a new leaf scalar with zero gradient.
    pub fn new(data: f64) -> Self {
        Self::with_op(data, Op::Leaf)
    }

    fn with_op(data: f64, op: Op) -> Self {
        Self(Rc::new(RefCell::new(ValueInner {
            data,
            grad: 0.0,
            op,
        })))
    }

    /// Returns the scalar's current forward value.
    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    /// Updates the scalar's forward value (e.g., during an optimizer step).
    pub fn set_data(&self, data: f64) {
        self.0.borrow_mut().data = data;
    }

    /// Returns the accumulated gradient with respect to the backward root.
    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    /// Overwrites the stored gradient.
    pub fn set_grad(&self, grad: f64) {
        self.0.borrow_mut().grad = grad;
    }

    /// Resets the stored gradient to zero.
    pub fn zero_grad(&self) {
        self.0.borrow_mut().grad = 0.0;
    }

    fn add_grad(&self, delta: f64) {
        self.0.borrow_mut().grad += delta;
    }

    /// Unique identity of this node within the computation graph.
    pub fn id(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    /// Human-readable label of the operation that created this node, if any.
    pub fn op_label(&self) -> Option<String> {
        match &self.0.borrow().op {
            Op::Leaf => None,
            Op::Add(_, _) => Some("+".to_string()),
            Op::Mul(_, _) => Some("*".to_string()),
            Op::Pow(_, exp) => Some(format!("**{exp}")),
            Op::Relu(_) => Some("ReLU".to_string()),
        }
    }

    /// Direct input nodes (`_prev`) in deterministic order.
    pub fn children(&self) -> Vec<Value> {
        match &self.0.borrow().op {
            Op::Leaf => Vec::new(),
            Op::Add(a, b) | Op::Mul(a, b) => {
                if a.id() == b.id() {
                    vec![a.clone()]
                } else {
                    vec![a.clone(), b.clone()]
                }
            }
            Op::Pow(a, _) | Op::Relu(a) => vec![a.clone()],
        }
    }

    /// Raises `self` to a constant `f64` exponent.
    pub fn powf(&self, exp: f64) -> Self {
        let out_data = self.data().powf(exp);
        Self::with_op(out_data, Op::Pow(self.clone(), exp))
    }

    /// Applies the Rectified Linear Unit (`max(0, x)`).
    pub fn relu(&self) -> Self {
        let x = self.data();
        let out_data = if x < 0.0 { 0.0 } else { x };
        Self::with_op(out_data, Op::Relu(self.clone()))
    }

    /// Runs reverse-mode automatic differentiation starting from `self`.
    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        build_topo(self, &mut visited, &mut topo);

        self.set_grad(1.0);
        for node in topo.into_iter().rev() {
            node.backward_step();
        }
    }

    fn backward_step(&self) {
        // Clone op and scalars first so no RefCell borrows overlap when a node appears twice.
        let (out_data, out_grad, op) = {
            let inner = self.0.borrow();
            (inner.data, inner.grad, inner.op.clone())
        };

        match op {
            Op::Leaf => {}
            Op::Add(lhs, rhs) => {
                lhs.add_grad(out_grad);
                rhs.add_grad(out_grad);
            }
            Op::Mul(lhs, rhs) => {
                let lhs_data = lhs.data();
                let rhs_data = rhs.data();
                lhs.add_grad(rhs_data * out_grad);
                rhs.add_grad(lhs_data * out_grad);
            }
            Op::Pow(base, exp) => {
                let base_data = base.data();
                base.add_grad(exp * base_data.powf(exp - 1.0) * out_grad);
            }
            Op::Relu(input) => {
                if out_data > 0.0 {
                    input.add_grad(out_grad);
                }
            }
        }
    }

    /// Renders the computation DAG rooted at `self` in Graphviz DOT format.
    pub fn to_dot(&self) -> String {
        let mut nodes = Vec::new();
        let mut seen = HashSet::new();
        let mut edges = BTreeSet::new();
        collect_graph(self, &mut seen, &mut nodes, &mut edges);

        let mut dot = String::from("digraph {\n    rankdir=LR;\n");
        for node in &nodes {
            let id = node.id();
            dot.push_str(&format!(
                "    \"{id}\" [label=\"{{ data {:.4} | grad {:.4} }}\", shape=record];\n",
                node.data(),
                node.grad()
            ));
            if let Some(op) = node.op_label() {
                dot.push_str(&format!("    \"{id}{op}\" [label=\"{op}\"];\n"));
                dot.push_str(&format!("    \"{id}{op}\" -> \"{id}\";\n"));
            }
        }

        for (src_id, dst_id, dst_op) in edges {
            dot.push_str(&format!("    \"{src_id}\" -> \"{dst_id}{dst_op}\";\n"));
        }
        dot.push_str("}\n");
        dot
    }
}

fn build_topo(node: &Value, visited: &mut HashSet<usize>, topo: &mut Vec<Value>) {
    if visited.insert(node.id()) {
        for child in node.children() {
            build_topo(&child, visited, topo);
        }
        topo.push(node.clone());
    }
}

fn collect_graph(
    node: &Value,
    seen: &mut HashSet<usize>,
    nodes: &mut Vec<Value>,
    edges: &mut BTreeSet<(usize, usize, String)>,
) {
    if seen.insert(node.id()) {
        nodes.push(node.clone());
        let dst_id = node.id();
        let dst_op = node.op_label().unwrap_or_default();
        for child in node.children() {
            edges.insert((child.id(), dst_id, dst_op.clone()));
            collect_graph(&child, seen, nodes, edges);
        }
    }
}

impl From<f64> for Value {
    fn from(data: f64) -> Self {
        Self::new(data)
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.0.borrow();
        write!(f, "Value(data={}, grad={})", inner.data, inner.grad)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl Sum<Value> for Value {
    fn sum<I: Iterator<Item = Value>>(iter: I) -> Self {
        iter.fold(Value::new(0.0), |acc, x| acc + x)
    }
}

impl<'a> Sum<&'a Value> for Value {
    fn sum<I: Iterator<Item = &'a Value>>(iter: I) -> Self {
        iter.fold(Value::new(0.0), |acc, x| acc + x)
    }
}

impl Neg for &Value {
    type Output = Value;

    fn neg(self) -> Value {
        self * -1.0
    }
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Value {
        (&self).neg()
    }
}

fn add_values(lhs: &Value, rhs: &Value) -> Value {
    Value::with_op(lhs.data() + rhs.data(), Op::Add(lhs.clone(), rhs.clone()))
}

fn mul_values(lhs: &Value, rhs: &Value) -> Value {
    Value::with_op(lhs.data() * rhs.data(), Op::Mul(lhs.clone(), rhs.clone()))
}

fn sub_values(lhs: &Value, rhs: &Value) -> Value {
    add_values(lhs, &(-rhs))
}

fn div_values(lhs: &Value, rhs: &Value) -> Value {
    mul_values(lhs, &rhs.powf(-1.0))
}

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $func:ident) => {
        impl $trait<&Value> for &Value {
            type Output = Value;
            fn $method(self, rhs: &Value) -> Value {
                $func(self, rhs)
            }
        }

        impl $trait<Value> for &Value {
            type Output = Value;
            fn $method(self, rhs: Value) -> Value {
                $func(self, &rhs)
            }
        }

        impl $trait<&Value> for Value {
            type Output = Value;
            fn $method(self, rhs: &Value) -> Value {
                $func(&self, rhs)
            }
        }

        impl $trait<Value> for Value {
            type Output = Value;
            fn $method(self, rhs: Value) -> Value {
                $func(&self, &rhs)
            }
        }

        impl $trait<f64> for &Value {
            type Output = Value;
            fn $method(self, rhs: f64) -> Value {
                $func(self, &Value::new(rhs))
            }
        }

        impl $trait<f64> for Value {
            type Output = Value;
            fn $method(self, rhs: f64) -> Value {
                $func(&self, &Value::new(rhs))
            }
        }

        impl $trait<&Value> for f64 {
            type Output = Value;
            fn $method(self, rhs: &Value) -> Value {
                $func(&Value::new(self), rhs)
            }
        }

        impl $trait<Value> for f64 {
            type Output = Value;
            fn $method(self, rhs: Value) -> Value {
                $func(&Value::new(self), &rhs)
            }
        }
    };
}

impl_binary_op!(Add, add, add_values);
impl_binary_op!(Sub, sub, sub_values);
impl_binary_op!(Mul, mul, mul_values);
impl_binary_op!(Div, div, div_values);
