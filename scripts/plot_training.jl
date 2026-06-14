using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using CairoMakie
using Statistics

println("Loading training data")
df = CSV.read("output/data/training_data.csv", DataFrame)
println("Loaded $(nrow(df)) epochs")

bin_size = 100_000
df.bin = div.(df.Epoch .- 1, bin_size) .+ 1
df.bin_label = df.bin .* bin_size

binned = combine(groupby(df, :bin_label),
    :Suma_nagrod => mean => :avg_reward,
)
sort!(binned, :bin_label)

println("Binned into $(nrow(binned)) groups of $(bin_size) epochs each")
println(binned)

println("\nGenerating plot")

set_theme!(Theme(
    fontsize=16,
    Axis=(
        xgridvisible=false,
        ygridvisible=true,
        ygridcolor=(:gray, 0.15),
    ),
))

fig = Figure(size=(1000, 550), backgroundcolor=:white)
ax = Axis(fig[1, 1],
    title="Postęp nauki — średnia nagroda co $(div(bin_size, 1000))k epok",
    xlabel="Epoka",
    ylabel="Średnia suma nagród",
)

x_vals = binned.bin_label ./ 1_000_000

lines!(ax, x_vals, binned.avg_reward,
    color=:dodgerblue, linewidth=2.5)
scatter!(ax, x_vals, binned.avg_reward,
    color=:dodgerblue, markersize=6)

ax.xlabel = "Epoka [mln]"

save("output/plots/training_plot.png", fig, px_per_unit=2)
