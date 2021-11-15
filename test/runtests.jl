using MLJBase: fit
using MLJTuringInterface
using Test
using Turing:
    I,
    MvNormal,
    Normal,
    NUTS,
    @model

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
    chns, _, _ = fit(tm, verbosity, X, y)

    @test MLJTuringInterface._parameter_mean(chns, :intercept) ≈ 0 atol=0.05
    @test MLJTuringInterface._parameter_mean(chns, :coef) ≈ 0.3 atol=0.05
end
