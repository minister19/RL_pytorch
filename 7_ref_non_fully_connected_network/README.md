### ref:

1. https://discuss.pytorch.org/t/custom-connections-in-neural-network-layers/3027
2. https://github.com/uchida-takumi/CustomizedLinear
3. https://discuss.pytorch.org/t/implement-selected-sparse-connected-neural-network/45517

### NOTES:

1. Weights being zero at the beginning doesn’t guarantee weights being zero throughout.
2. You cant prevent the weights from changing during gradient descent. However, you can introduce a parameter called mask, which multiplies a mask with the weights, and use that to do the forward pass and the backward pass.
