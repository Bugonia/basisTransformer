# General Residual Write Economy

## Core Generalization

The Transformer case is not isolated. Many residual networks can be written as

```text
x_{l+1} = x_l + g_l(x_l)
        = x_l + B_l c_l(x_l)
```

whenever the residual branch ends in a learned linear map. The final map
provides a learned write dictionary, while the preceding nonlinear computation
generates state-dependent coefficients.

## Examples

- Residual MLP:
  - final linear layer columns are write directions;
  - hidden activations are coefficients.
- ResNet bottleneck block:
  - final convolution output channels act as spatially shared write
    dictionaries;
  - earlier convolutions, normalization, and nonlinearities generate
    location-dependent coefficients.
- Transformer FFN:
  - down-projection columns are token-local write directions;
  - MLP activations are coefficients.
- Transformer Attention:
  - output projection columns are write directions;
  - attention routing and value coordinates generate coefficients.

## Why Transformers Remain the Paper 1 Testbed

Transformers are the cleanest first testbed because one block has multiple
heterogeneous residual writers:

```text
Attention write basis + FFN write basis
```

This lets us experimentally separate:

- direct write access;
- coefficient-only modulation;
- same-layer coefficient coupling;
- sublayer ordering.

In a standard ResNet or residual MLP, the residual branch often has one obvious
final output dictionary per block, so the same theory applies but the
architecture does not naturally expose the same Attention/FFN write-right
contrast.

## Reviewer-Facing Boundary

Use this claim:

> We propose a general basis/coefficient view of residual updates and test it in
> Transformers, where multiple heterogeneous residual writers make direct write
> access experimentally separable.

Avoid this overclaim:

> We have experimentally explained residual connections in all deep networks.

## Follow-Up Experiments

- Residual MLP:
  - compare blocks with one direct write versus nested coefficient-only branch.
- ResNet:
  - modify bottleneck blocks so one convolutional branch modulates coefficients
    without direct output-channel write access.
- Vision Transformer:
  - repeat Attention/MLP write-access ablations on image classification.
