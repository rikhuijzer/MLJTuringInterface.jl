module MLJTuringInterface

import MLJModelInterface:
    fit

using MLJModelInterface:
    Probabilistic
using Turing:
    MCMCThreads,
    DynamicPPL,
    chainscat,
    predict,
    sample

mutable struct TuringModel <: Probabilistic
    model::DynamicPPL.Model
    n_chains::Int
    n_samples::Int
    sampler::Any
    multithreaded::Bool
end

function fit(tm::TuringModel, verbosity::Int, X, y)
    chns = if tm.multithreaded
        sample_func(c) = sample(tm.model, tm.sampler, tm.n_samples)
        mapreduce(sample_func, chainscat, 1:tm.n_chains)
    else
        sample(tm.model, tm.sampler, MCMCThreads(), tm.n_samples, tm.n_chains)
    end
    cache = nothing
    report = nothing

    return fitresult, cache, report
end

end # module
