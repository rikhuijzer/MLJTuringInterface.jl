let
    name = "Turing"
    uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
    url = "https://github.com/TuringLang/Turing.jl"
    julia = true
    license = "MIT"
    is_wrapper = false
    metadata_pkg(TuringModel; name, uuid, url, julia, license, is_wrapper)
end

let
    input = Table(Continuous)
    target = AbstractVector{Continuous}
    descr = "Bayesian model defined via Turing"
    path = "$PKG.TuringModel"
    metadata_model(TuringModel; input, target, descr, path)
end
