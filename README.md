# MLJ.jl <> Turing.jl

Interface for [Turing](https://github.com/TuringLang/Turing.jl) models with [MLJ](https://github.com/alan-turing-institute/MLJ.jl).

## Usage

For example, to create a `machine` for a linear regression, use:

```julia
@model function linear_regression(X, y)
    [...]
end

model = linear_regression
n_samples = 100
sampler = NUTS()
tm = TuringModel(model, n_samples, sampler)
mach = machine(tm, X, y)
```

This machine can then be used like normally in `MLJ`.

See the tests for more detailed examples.
