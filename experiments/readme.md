# Experiments with PyTorch

## MNIST Test Suite

You can run any of the following command.
Each of them is a stand-alone experiment that takes MNIST images as input.

``` bash
$ python experiments/mnist/sum_2.py
$ python experiments/mnist/sum_3.py
$ python experiments/mnist/sum_4.py
$ python experiments/mnist/add_sub.py
$ python experiments/mnist/mult_2.py
$ python experiments/mnist/sort_2/run.py
```

## Pathfinder

There are two tasks in pathfinder: `pathfinder_32` and `pathfinder_128`.
`pathfinder_32` contains images of size 32 x 32, and `pathfinder_128` contains images
of size 128 x 128.
As you might imagine, `pathfinder_128` is significantly more difficult than
`pathfinder_32`.

You can run the following commands to do experiment on these two tasks:

``` bash
$ python experiments/pathfinder/32/run.py
$ python experiments/pathfinder/128/run_with_cnn.py
```

## Handwritten Formula (HWF)

You can run the following command to start the training of HWF

``` bash
$ python experiments/hwf/run_with_unique_logical_parser.py --provenance diffminmaxprob
```
