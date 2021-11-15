module MLJTuringInterface

export TuringModel

import MLJModelInterface

using MLJModelInterface: Probabilistic
using Statistics: mean
using Turing:
    MCMCThreads,
    DynamicPPL,
    chainscat,
    predict,
    sample

const MMI = MLJModelInterface

mutable struct TuringModel <: Probabilistic
    model::Any
    n_samples::Int
    sampler::Any
    n_chains::Int
    multithreaded::Bool
end

function TuringModel(model, n_samples, sampler; n_chains=3, multithreaded=true)
    return TuringModel(model, n_samples, sampler, n_chains, multithreaded)
end

function _parameter_mean(chns, parameter::Symbol)
    values = collect(Iterators.flatten(chns[parameter]))
    return mean(values)
end

function MMI.fit(tm::TuringModel, verbosity::Int, X, y)
    model = tm.model(X, y)
    chns = if tm.multithreaded
        sample_func(c) = sample(model, tm.sampler, tm.n_samples)
        mapreduce(sample_func, chainscat, 1:tm.n_chains)
    else
        sample(model, tm.sampler, MCMCThreads(), tm.n_samples, tm.n_chains)
    end
    fitresult = chns
    cache = nothing
    report = nothing

    return fitresult, cache, report
end

function MMI.predict(tm::TuringModel, fitresult, Xnew)
    chns = predict(tm.model, fitresult)
    y = mean(Array(group(chns, :y)); dims=1)
    return y
end

end # module
