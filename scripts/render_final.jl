using Pkg
Pkg.activate(".")

include("src/Environment.jl")
include("src/RLWrapper.jl")
include("src/Training.jl")
include("src/Visualization.jl")

using .Visualization

models = fill("model_checkpoint.bson", 10)

for n in 1:12
    println("Rendering $n car(s)")
    create_timelapse(num_cars=n, max_steps=500, model_files=models, output_file="output/videos/final_model_$(n)cars.mp4")
end

println("done")
