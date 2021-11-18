"""
    TuringModel <: Probabilistic

Fields:
- `model`: a Turing model function; define it via `@model func(X, y)`
- `n_samples`
- `sampler`
- `renamer`: Can be used to rename the MCMCChains parameter names.
    Defaults to `nothing`.
    If `renamer isa Dict`, then names are dropped from the data `X`.
- `n_chains`
- `multithreaded`: Defaults to `true`; this can be set to false when doing threaded evaluations via MLJ.
"""
mutable struct TuringModel <: Probabilistic
    model::Function
    n_samples::Int
    sampler::Any
    renamer::Union{Nothing,Dict}
    n_chains::Int
    multithreaded::Bool
end

function TuringModel(model, n_samples, sampler; renamer=nothing, n_chains=3, multithreaded=true)
    return TuringModel(model, n_samples, sampler, renamer, n_chains, multithreaded)
end

function _parameter_mean(chns, parameter)
    values = collect(Iterators.flatten(chns[parameter]))
    return mean(values)
end

"""
    _simplify_data(X) -> Matrix

Convert `X` to a Matrix since this usually works better with Turing.jl models.
Assumes that `X` satisfies the Tables interface.
This code is partially based on `Base.Matrix` in DataFrames.jl.
"""
function _simplify_data(X)::Matrix
    nrows = length(Tables.rows(X))
    ncols = length(Tables.columns(X))
    T = reduce(promote_type, (eltype(v) for v in Tables.columns(X)), init=Union{})
    out = Matrix{T}(undef, nrows, ncols)
    for (i, row) in enumerate(Tables.rows(X))
        for (j, col) in enumerate(Tables.columnnames(row))
            out[i, j] = Tables.getcolumn(row, col)
        end
    end
    return out
end

function MMI.fit(tm::TuringModel, verbosity::Int, X, y)
    renamed_X = isnothing(tm.renamer) ? X : _simplify_data(X)
    model = tm.model(renamed_X, y)
    chns = if tm.multithreaded
        sample_func(c) = sample(model, tm.sampler, tm.n_samples)
        mapreduce(sample_func, chainscat, 1:tm.n_chains)
    else
        sample(model, tm.sampler, MCMCThreads(), tm.n_samples, tm.n_chains)
    end
    renamed_chns = isnothing(tm.renamer) ? chns : replacenames(chns, tm.renamer)
    fitresult = renamed_chns
    cache = nothing
    report = nothing

    return fitresult, cache, report
end

function MMI.predict(tm::TuringModel, fitresult, Xnew)
    model = tm.model(Xnew, missing)
    chns = predict(model, fitresult)
    y_matrix = mean(Array(group(chns, :y)); dims=1)
    y = collect(Iterators.flatten(y_matrix))
    return y
end

