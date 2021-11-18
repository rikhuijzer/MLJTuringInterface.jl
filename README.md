# MLJ.jl <> Turing.jl

Interface for [Turing](https://github.com/TuringLang/Turing.jl) models with [MLJ](https://github.com/alan-turing-institute/MLJ.jl).

## Usage

For example, to create a `machine` for a linear regression, use something like:

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

where `X` and `y` are your features and predictor variables respectively.
Next, this machine can be used like normally in `MLJ`.

See the tests for more detailed examples.

### Named data

Usually, MLJ interface models can take any object which satisfies the `Tables.jl` interface.
Turing.jl does so too, but this will usually not work out when the Turing model uses broadcasting.

To work around this, this package allows defining a `renamer::Dict` which is a mapping.
When this `renamer` is set, `X` is transformed to a `Matrix` before being passed to Turing.
After fitting the model, the parameter names are updated based on the renames in `renamer`.

For example, when fitting a linear model with something like

```julia
@model function linear_regression(X::Matrix, y)
    σ₂ ~ truncated(Normal(0, 100), 0, Inf)

    intercept ~ Normal(0, 0.4)

    n_features = size(X, 2)
    coef ~ MvNormal(n_features, 0.4)

    mu = intercept .+ X * coef
    y ~ MvNormal(mu, sqrt(σ₂))
end;
```

then the output contains `coef[1]`, `coef[2]`, ..., `coef[n]`.

To rename this to the keys of a namedtuple, use:

```julia
[...]
renamer = Dict(["coef[$i]" => "coef[$key]" for (i, key) in enumerate(keys(X))])
tm = TuringModel(model, n_samples, sampler; renamer)
```
