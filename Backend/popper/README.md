# POPPER IS STILL A MAJOR WORK-IN-PROGRESS. BE PREPARED FOR BUGS, DEMONS, AND BACKWARDS-BREAKING CHANGES!


# Popper

Popper is an inductive logic programming (ILP) system.
If you use Popper for research, please cite the paper [learning programs by learning from failures](https://arxiv.org/abs/2005.02259).



## Requirements

[SWI-Prolog](https://www.swi-prolog.org)

[Clingo 5.5.0](https://potassco.org/clingo/)

[pyswip](https://pypi.org/project/pyswip/)


# Usage

You can run Popper like so:
```
python popper.py examples/dropk/
f(A,B,C) :- one(B),tail(A,C).
f(A,B,C) :- tail(A,E),decrement(B,D),f(E,D,C).
```
Take a look at the examples folder for examples.

# Popper settings

By default, Popper does not test all the examples during the testing stage. To do so, call Popper with the `--test-all` flag.
