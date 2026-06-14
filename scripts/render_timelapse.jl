using Pkg
Pkg.activate(".")
include("src/Environment.jl")
include("src/RLWrapper.jl")
include("src/Visualization.jl")
using .Visualization

checkpoint_epochs = [1, 50000, 100000, 200000, 400000, 600000, 800000,
    1000000, 1500000, 2000000, 2500000, 3000000,
    3500000, 4000000]

models = ["model_checkpoint_epoch_$(e).bson" for e in checkpoint_epochs]

println("Creating training timelapse with $(length(models)) checkpoints")
create_timelapse(num_cars=8, max_steps=500, model_files=models, output_file="output/videos/traffic_simulation_timelapse.mp4")
println("Timelapse done")
