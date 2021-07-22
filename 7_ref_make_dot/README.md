### ref:

1. https://graphviz.readthedocs.io/en/stable/manual.html
2. https://github.com/szagoruyko/pytorchviz

### NOTES:

1. Need to install graphviz first and add `C:\Program Files\Graphviz\bin\;` to OS user/system path.
2. make_dot expects a variable (i.e., tensor with grad_fn), not the model itself. (ref: https://stackoverflow.com/questions/52468956/how-do-i-visualize-a-net-in-pytorch)
