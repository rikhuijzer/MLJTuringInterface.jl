module MLJTuringInterface

export TuringModel

import MLJModelInterface

using MLJModelInterface:
    Probabilistic,
    Table,
    Continuous,
    metadata_model,
    metadata_pkg
using Statistics: mean
using Tables
using Turing:
    MCMCThreads,
    DynamicPPL,
    chainscat,
    group,
    predict,
    sample

const MMI = MLJModelInterface
const PKG = "MLJTuringInterface"

include("interface.jl")
include("metadata.jl")

end # module
