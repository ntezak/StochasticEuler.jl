# StochasticEuler

[![Build Status](https://travis-ci.org/ntezak/StochasticEuler.jl.svg?branch=master)](https://travis-ci.org/ntezak/StochasticEuler.jl)

`StochasticEuler.jl` is a lightweight [Julia][4] package for integrating real and complex valued high dimensional [stochastic differential equations][2] (supporting multi-dimensional noises). Both Ito and Stratonovich SDEs are supported. It also features some additional tools for verifying path-wise stochastic convergence.

The integration method is a fixed stepsize implicit Euler-Heun (Euler-Mayurama) for a Stratonivich (Ito) SDE.
These algorithms are comparable to those published under [SDELab][1].
All relevant functions have docstrings that are accessible via the `?` prefix in the REPL or jupyter notebook.

Please check out the [example Jupyter notebook][3] for some use cases until a full documentation has been written up.

  [1]: http://doi.org/10.1016/j.cam.2006.05.037 "Gilsing & Shardlow (2007). SDELab: A package for solving stochastic differential equations in MATLAB"
  [2]: https://en.wikipedia.org/wiki/Stochastic_differential_equation "Wikipedia: Stochastic Differential Equations"
  [3]: https://github.com/ntezak/StochasticEuler.jl/blob/master/examples/introduction.ipynb "Introductory jupyter notebook"
  [4]: http://julialang.org/ "The Julia Language"
