# Repeatability package for AISoLA 2023

This is the repeatability evaluation (RE) package for the paper *The inverse problem for neural networks* presented
at AISoLA 2023.
To cite the work, you can use:

```
@inproceedings{ForetsS24,
  author       = {Marcelo Forets and
                  Christian Schilling},
  title        = {The inverse problem for neural networks},
  booktitle    = {{(A)ISoLA}},
  series       = {LNCS},
  volume       = {14380},
  publisher    = {Springer},
  year         = {2024},
  url          = {https://doi.org/10.1007/978-3-031-46002-9\_14},
  doi          = {10.1007/978-3-031-46002-9\_14}
}
```


## How to use

First install the Julia compiler following the instructions [here](http://julialang.org/downloads).
Once you have installed Julia, open a terminal in the `examples/` folder and execute

```shell
$ julia --project=. run_all.jl
```

to run all experiments. Alternatively, each experiment can be run individually. Check the file
`run_all.jl` to identify the relevant scripts.

Each experiment creates plot files in the current folder.


## Parabola experiment

The neural network in `parabola2d/parabola2d.network` was created with the script
`parabola2d/parabola2d_train.jl`. This script results in different neural networks every time it is
run.
