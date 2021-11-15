using MLJBase:
    fit,
    predict
using MLJTuringInterface
using Test
using Turing:
    I,
    MvNormal,
    Normal,
    NUTS,
    @model

function _allclose(A, B; atol=0.1)
    @assert length(A) == length(B)
    for (i, t) in enumerate(zip(A, B))
        a, b = t
        if !isapprox(a, b; atol)
            error("!($a ≈ $b) at position $i")
        end
    end
    return true
end

@testset "interface" begin
    # Using centered data to speed up sampling.
    X = (; X = collect(-3:3))
    y = collect(-0.9:0.3:0.9)

    @model function linear_regression(X, y)
        σ = 0.2
        intercept ~ Normal(0, σ)
        coef ~ Normal(0, σ)

        mu = intercept .+ X.X * coef
        y ~ MvNormal(mu, σ^2 * I)
    end

    model = linear_regression
    n_samples = 100
    sampler = NUTS()
    tm = TuringModel(model, n_samples, sampler)

    verbosity = 1
    fitresult, _, _ = fit(tm, verbosity, X, y)
    chns = fitresult

    @test MLJTuringInterface._parameter_mean(chns, :intercept) ≈ 0 atol=0.05
    @test MLJTuringInterface._parameter_mean(chns, :coef) ≈ 0.3 atol=0.05

    predictions = predict(tm, chns, X)
    @test _allclose(y, Iterators.flatten(predictions))
end
