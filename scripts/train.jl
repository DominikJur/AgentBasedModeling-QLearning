using Pkg
Pkg.activate(".")

include("src/Environment.jl")
include("src/RLWrapper.jl")
include("src/Training.jl")

using .Training

train_model(epochs=4000000, max_steps=500, num_cars=10)
