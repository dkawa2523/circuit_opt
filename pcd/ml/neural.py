"""Small deterministic NumPy neural models for the AC corpus comparison.

These are deliberately fixed reference models, not a tuning framework.  The
MLP uses flat graph/context features.  The GNN learns separate messages for
positive and negative component terminals in both directions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .graph_encoding import GraphBatch

_HIDDEN = 24
_GRAPH_HIDDEN = 16
_EPOCHS = 300
_LEARNING_RATE = 0.01
_L2 = 1e-4
_GRADIENT_LIMIT = 5.0


@dataclass(frozen=True)
class NeuralFit:
    """Predictions plus enough training evidence to detect numerical failure."""

    predictions: np.ndarray
    initial_loss: float
    final_loss: float
    parameter_count: int


@dataclass
class _AdamState:
    first: dict[str, np.ndarray]
    second: dict[str, np.ndarray]


def fit_mlp(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    prediction_features: np.ndarray,
    *,
    seed: int,
    epochs: int = _EPOCHS,
) -> NeuralFit:
    """Fit a fixed two-hidden-layer MLP and predict standardized targets."""

    _validate_supervised_arrays(train_features, train_targets, prediction_features)
    rng = np.random.default_rng(seed)
    inputs = train_features.shape[1]
    outputs = train_targets.shape[1]
    parameters = {
        "w1": _weight(rng, inputs, _HIDDEN),
        "b1": np.zeros(_HIDDEN),
        "w2": _weight(rng, _HIDDEN, _HIDDEN),
        "b2": np.zeros(_HIDDEN),
        "wo": _weight(rng, _HIDDEN, outputs),
        "bo": np.zeros(outputs),
    }
    state = _adam_state(parameters)
    initial = _mse(_mlp_forward(train_features, parameters)[0], train_targets)
    for step in range(1, _positive_epochs(epochs) + 1):
        predicted, cache = _mlp_forward(train_features, parameters)
        gradients = _mlp_gradients(parameters, cache, predicted, train_targets)
        _adam_update(parameters, gradients, state, step)
    final = _mse(_mlp_forward(train_features, parameters)[0], train_targets)
    predictions = _mlp_forward(prediction_features, parameters)[0]
    _require_finite_training(initial, final, predictions)
    return NeuralFit(predictions, initial, final, _parameter_count(parameters))


def fit_relational_gnn(
    train_graphs: GraphBatch,
    train_context: np.ndarray,
    train_targets: np.ndarray,
    prediction_graphs: GraphBatch,
    prediction_context: np.ndarray,
    *,
    seed: int,
    epochs: int = _EPOCHS,
) -> NeuralFit:
    """Fit a two-step relation-specific component/net message-passing model."""

    _validate_graph_arrays(train_graphs, train_context, train_targets, prediction_graphs, prediction_context)
    rng = np.random.default_rng(seed)
    inputs = train_graphs.node_features.shape[2]
    outputs = train_targets.shape[1]
    head_inputs = 4 * _GRAPH_HIDDEN + train_context.shape[1]
    parameters = {
        "input_w": _weight(rng, inputs, _GRAPH_HIDDEN),
        "input_b": np.zeros(_GRAPH_HIDDEN),
        "l1_self": _weight(rng, _GRAPH_HIDDEN, _GRAPH_HIDDEN),
        "l1_rel": np.stack([_weight(rng, _GRAPH_HIDDEN, _GRAPH_HIDDEN) for _ in range(4)]),
        "l1_b": np.zeros(_GRAPH_HIDDEN),
        "l2_self": _weight(rng, _GRAPH_HIDDEN, _GRAPH_HIDDEN),
        "l2_rel": np.stack([_weight(rng, _GRAPH_HIDDEN, _GRAPH_HIDDEN) for _ in range(4)]),
        "l2_b": np.zeros(_GRAPH_HIDDEN),
        "head_w": _weight(rng, head_inputs, _HIDDEN),
        "head_b": np.zeros(_HIDDEN),
        "out_w": _weight(rng, _HIDDEN, outputs),
        "out_b": np.zeros(outputs),
    }
    state = _adam_state(parameters)
    initial = _mse(_gnn_forward(train_graphs, train_context, parameters)[0], train_targets)
    for step in range(1, _positive_epochs(epochs) + 1):
        predicted, cache = _gnn_forward(train_graphs, train_context, parameters)
        gradients = _gnn_gradients(parameters, cache, predicted, train_targets)
        _adam_update(parameters, gradients, state, step)
    final = _mse(_gnn_forward(train_graphs, train_context, parameters)[0], train_targets)
    predictions = _gnn_forward(prediction_graphs, prediction_context, parameters)[0]
    _require_finite_training(initial, final, predictions)
    return NeuralFit(predictions, initial, final, _parameter_count(parameters))


def _validate_supervised_arrays(train_x: np.ndarray, train_y: np.ndarray, predict_x: np.ndarray) -> None:
    _require_finite_matrix(train_x, "MLP training features")
    _require_finite_matrix(train_y, "MLP training targets")
    _require_finite_matrix(predict_x, "MLP prediction features")
    if not len(train_x) or not len(predict_x) or len(train_x) != len(train_y):
        raise ValueError("MLP requires non-empty aligned training and prediction rows")
    if train_x.shape[1] != predict_x.shape[1] or not train_x.shape[1] or not train_y.shape[1]:
        raise ValueError("MLP feature and target widths must be non-empty and consistent")


def _validate_graph_arrays(
    train_graphs: GraphBatch,
    train_context: np.ndarray,
    train_targets: np.ndarray,
    prediction_graphs: GraphBatch,
    prediction_context: np.ndarray,
) -> None:
    _require_finite_matrix(train_context, "GNN training context")
    _require_finite_matrix(prediction_context, "GNN prediction context")
    _require_finite_matrix(train_targets, "GNN training targets")
    if not len(train_context) or not len(prediction_context) or len(train_context) != len(train_targets):
        raise ValueError("GNN requires non-empty aligned training and prediction rows")
    if train_context.shape[1] != prediction_context.shape[1] or not train_targets.shape[1]:
        raise ValueError("GNN context widths must match and target width must be non-empty")
    if len(train_graphs.node_features) != len(train_context):
        raise ValueError("training graphs and context rows must align")
    if len(prediction_graphs.node_features) != len(prediction_context):
        raise ValueError("prediction graphs and context rows must align")
    if train_graphs.node_features.shape[2] != prediction_graphs.node_features.shape[2]:
        raise ValueError("training and prediction graph feature widths must match")


def _require_finite_matrix(value: np.ndarray, name: str) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _mlp_forward(features: np.ndarray, parameters: dict[str, np.ndarray]) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    hidden1 = np.tanh(features @ parameters["w1"] + parameters["b1"])
    hidden2 = np.tanh(hidden1 @ parameters["w2"] + parameters["b2"])
    predicted = hidden2 @ parameters["wo"] + parameters["bo"]
    return predicted, (features, hidden1, hidden2)


def _mlp_gradients(
    parameters: dict[str, np.ndarray],
    cache: tuple[np.ndarray, ...],
    predicted: np.ndarray,
    targets: np.ndarray,
) -> dict[str, np.ndarray]:
    features, hidden1, hidden2 = cache
    output_gradient = _mse_gradient(predicted, targets)
    hidden2_gradient = (output_gradient @ parameters["wo"].T) * (1.0 - hidden2**2)
    hidden1_gradient = (hidden2_gradient @ parameters["w2"].T) * (1.0 - hidden1**2)
    return {
        "w1": features.T @ hidden1_gradient + _L2 * parameters["w1"],
        "b1": hidden1_gradient.sum(axis=0),
        "w2": hidden1.T @ hidden2_gradient + _L2 * parameters["w2"],
        "b2": hidden2_gradient.sum(axis=0),
        "wo": hidden2.T @ output_gradient + _L2 * parameters["wo"],
        "bo": output_gradient.sum(axis=0),
    }


def _gnn_forward(
    graphs: GraphBatch,
    context: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, object]]:
    mask = graphs.node_mask[..., np.newaxis]
    initial = np.tanh(graphs.node_features @ parameters["input_w"] + parameters["input_b"]) * mask
    layer1, cache1 = _relation_forward(
        initial,
        graphs.adjacency,
        mask,
        parameters["l1_self"],
        parameters["l1_rel"],
        parameters["l1_b"],
    )
    layer2, cache2 = _relation_forward(
        layer1,
        graphs.adjacency,
        mask,
        parameters["l2_self"],
        parameters["l2_rel"],
        parameters["l2_b"],
    )
    pooled, pool_cache = _pool_graphs(graphs, layer2)
    joined = np.concatenate([pooled, context], axis=1)
    hidden = np.tanh(joined @ parameters["head_w"] + parameters["head_b"])
    predicted = hidden @ parameters["out_w"] + parameters["out_b"]
    cache: dict[str, object] = {
        "initial": initial,
        "mask": mask,
        "cache1": cache1,
        "cache2": cache2,
        "pool": pool_cache,
        "joined": joined,
        "hidden": hidden,
        "node_features": graphs.node_features,
    }
    return predicted, cache


def _relation_forward(
    hidden: np.ndarray,
    adjacency: np.ndarray,
    mask: np.ndarray,
    self_weight: np.ndarray,
    relation_weight: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    messages = np.einsum("brij,bjh->brih", adjacency, hidden)
    combined = hidden @ self_weight + np.einsum("brih,rhk->bik", messages, relation_weight) + bias
    output = np.tanh(combined) * mask
    return output, (hidden, adjacency, mask, messages, output)


def _pool_graphs(graphs: GraphBatch, hidden: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    component_count = np.maximum(graphs.component_mask.sum(axis=1, keepdims=True), 1.0)
    component = np.einsum("bn,bnh->bh", graphs.component_mask, hidden) / component_count
    source = np.einsum("bn,bnh->bh", graphs.source_mask, hidden)
    load = np.einsum("bn,bnh->bh", graphs.load_mask, hidden)
    ground = np.einsum("bn,bnh->bh", graphs.ground_mask, hidden)
    return (
        np.concatenate([component, source, load, ground], axis=1),
        (graphs.component_mask, graphs.source_mask, graphs.load_mask, graphs.ground_mask, component_count),
    )


def _gnn_gradients(
    parameters: dict[str, np.ndarray],
    cache: dict[str, object],
    predicted: np.ndarray,
    targets: np.ndarray,
) -> dict[str, np.ndarray]:
    hidden = _array(cache["hidden"])
    joined = _array(cache["joined"])
    output_gradient = _mse_gradient(predicted, targets)
    hidden_gradient = (output_gradient @ parameters["out_w"].T) * (1.0 - hidden**2)
    joined_gradient = hidden_gradient @ parameters["head_w"].T
    graph_gradient = joined_gradient[:, : 4 * _GRAPH_HIDDEN]
    layer2_gradient = _pool_backward(_tuple_arrays(cache["pool"]), graph_gradient)
    layer1_gradient, layer2_gradients = _relation_backward(
        _tuple_arrays(cache["cache2"]),
        layer2_gradient,
        parameters["l2_self"],
        parameters["l2_rel"],
    )
    initial_gradient, layer1_gradients = _relation_backward(
        _tuple_arrays(cache["cache1"]),
        layer1_gradient,
        parameters["l1_self"],
        parameters["l1_rel"],
    )
    initial = _array(cache["initial"])
    mask = _array(cache["mask"])
    input_gradient = initial_gradient * (1.0 - initial**2) * mask
    node_features = _array(cache["node_features"])
    gradients = {
        "input_w": np.einsum("bnf,bnh->fh", node_features, input_gradient) + _L2 * parameters["input_w"],
        "input_b": input_gradient.sum(axis=(0, 1)),
        "l1_self": layer1_gradients[0] + _L2 * parameters["l1_self"],
        "l1_rel": layer1_gradients[1] + _L2 * parameters["l1_rel"],
        "l1_b": layer1_gradients[2],
        "l2_self": layer2_gradients[0] + _L2 * parameters["l2_self"],
        "l2_rel": layer2_gradients[1] + _L2 * parameters["l2_rel"],
        "l2_b": layer2_gradients[2],
        "head_w": joined.T @ hidden_gradient + _L2 * parameters["head_w"],
        "head_b": hidden_gradient.sum(axis=0),
        "out_w": hidden.T @ output_gradient + _L2 * parameters["out_w"],
        "out_b": output_gradient.sum(axis=0),
    }
    return gradients


def _relation_backward(
    cache: tuple[np.ndarray, ...],
    output_gradient: np.ndarray,
    self_weight: np.ndarray,
    relation_weight: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    hidden, adjacency, mask, messages, output = cache
    combined_gradient = output_gradient * (1.0 - output**2) * mask
    self_gradient = np.einsum("bnh,bnk->hk", hidden, combined_gradient)
    relation_gradient = np.einsum("brih,bik->rhk", messages, combined_gradient)
    message_gradient = np.einsum("bik,rhk->brih", combined_gradient, relation_weight)
    hidden_gradient = combined_gradient @ self_weight.T
    hidden_gradient += np.einsum("brij,brih->bjh", adjacency, message_gradient)
    return hidden_gradient, (self_gradient, relation_gradient, combined_gradient.sum(axis=(0, 1)))


def _pool_backward(cache: tuple[np.ndarray, ...], gradient: np.ndarray) -> np.ndarray:
    component, source, load, ground, component_count = cache
    chunks = np.split(gradient, 4, axis=1)
    result = component[..., np.newaxis] * chunks[0][:, np.newaxis, :] / component_count[..., np.newaxis]
    result += source[..., np.newaxis] * chunks[1][:, np.newaxis, :]
    result += load[..., np.newaxis] * chunks[2][:, np.newaxis, :]
    result += ground[..., np.newaxis] * chunks[3][:, np.newaxis, :]
    return result


def _adam_state(parameters: dict[str, np.ndarray]) -> _AdamState:
    return _AdamState(
        {name: np.zeros_like(value) for name, value in parameters.items()},
        {name: np.zeros_like(value) for name, value in parameters.items()},
    )


def _adam_update(
    parameters: dict[str, np.ndarray],
    gradients: dict[str, np.ndarray],
    state: _AdamState,
    step: int,
) -> None:
    _clip_gradients(gradients)
    for name, parameter in parameters.items():
        gradient = gradients[name]
        state.first[name] = 0.9 * state.first[name] + 0.1 * gradient
        state.second[name] = 0.999 * state.second[name] + 0.001 * gradient**2
        first = state.first[name] / (1.0 - 0.9**step)
        second = state.second[name] / (1.0 - 0.999**step)
        parameter -= _LEARNING_RATE * first / (np.sqrt(second) + 1e-8)


def _clip_gradients(gradients: dict[str, np.ndarray]) -> None:
    norm = math.sqrt(sum(float(np.sum(gradient**2)) for gradient in gradients.values()))
    if norm > _GRADIENT_LIMIT:
        scale = _GRADIENT_LIMIT / norm
        for gradient in gradients.values():
            gradient *= scale


def _weight(rng: np.random.Generator, inputs: int, outputs: int) -> np.ndarray:
    scale = math.sqrt(2.0 / (inputs + outputs))
    return rng.normal(0.0, scale, size=(inputs, outputs))


def _mse(predicted: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((predicted - targets) ** 2))


def _mse_gradient(predicted: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return 2.0 * (predicted - targets) / predicted.size


def _positive_epochs(epochs: int) -> int:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    return epochs


def _parameter_count(parameters: dict[str, np.ndarray]) -> int:
    return sum(parameter.size for parameter in parameters.values())


def _require_finite_training(initial: float, final: float, predictions: np.ndarray) -> None:
    if not math.isfinite(initial) or not math.isfinite(final) or not np.isfinite(predictions).all():
        raise ValueError("neural model produced non-finite training results")


def _array(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):  # pragma: no cover - internal cache invariant
        raise TypeError("neural cache entry is not an array")
    return value


def _tuple_arrays(value: object) -> tuple[np.ndarray, ...]:
    if not isinstance(value, tuple) or not all(isinstance(item, np.ndarray) for item in value):
        raise TypeError("neural cache entry is not an array tuple")  # pragma: no cover - internal invariant
    return value
