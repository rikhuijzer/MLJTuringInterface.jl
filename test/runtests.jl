using MLJBase:
    CV,
    evaluate!,
    fit,
    fit!,
    machine,
    predict,
    rms
using MLJTuringInterface
using Test
using Turing:
    Diagonal,
    I,
    MvNormal,
    Normal,
    NUTS,
    @model

"""
    _allclose(A, B; atol=0.1)

Helper function to return whether the elements in `A` and `B` are approximately equal.
Also throws an informative error if the elements are not approximately equal.
"""
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

@testset "turinginterface" begin
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

    @testset "interface" begin
        verbosity = 1
        fitresult, _, _ = fit(tm, verbosity, X, y)
        chns = fitresult

        @test MLJTuringInterface._parameter_mean(chns, :intercept) ≈ 0 atol=0.05
        @test MLJTuringInterface._parameter_mean(chns, :coef) ≈ 0.3 atol=0.05

        predictions = predict(tm, chns, X)
        @test _allclose(y, predictions)
    end

    @testset "cross-validation" begin
        mach = machine(tm, X, y)
        measure = rms
        nfolds = 2
        resampling = CV(; nfolds)
        result = evaluate!(mach; measure, resampling)
        # Smoke test.
        @test length(result.fitted_params_per_fold) == nfolds
    end
end

@testset "named_data" begin
    A = collect(-3:3)
    n = length(A)
    B = fill(1, n)
    X = (; A, B)
    y = collect(-0.9:0.3:0.9)

    @model function multivariate_regression(X, y)
        σ = 0.2
        intercept ~ Normal(0, σ)
        d = size(X, 2)
        coef ~ MvNormal(Diagonal(fill(σ ^ 2, d)))

        mu = intercept .+ X * coef
        y ~ MvNormal(mu, σ^2 * I)
    end

    model = multivariate_regression
    n_samples = 100
    sampler = NUTS()
    renamer = Dict(["coef[$i]" => "coef[$key]" for (i, key) in enumerate(keys(X))])
    tm = TuringModel(model, n_samples, sampler; renamer)
    mach = machine(tm, X, y)
    fit!(mach)

    chns = mach.fitresult
    @test MLJTuringInterface._parameter_mean(chns, "coef[A]") ≈ 0.3 atol=0.05
end
