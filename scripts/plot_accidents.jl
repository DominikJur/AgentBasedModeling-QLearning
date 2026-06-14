using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using CairoMakie
using Statistics

all_rows = CSV.read("output/data/accident_analysis_raw.csv", DataFrame)
println("Loaded $(nrow(all_rows)) rows")

const CAR_RANGE = 1:12

summary_df = combine(groupby(all_rows, :num_cars),
    :num_accidents => mean => :avg_accidents,
    :num_accidents => sum => :total_accidents,
    :num_accidents => maximum => :max_accidents,
    :any_accident => sum => :sims_with_accident,
    :any_accident => mean => :accident_probability
)

CSV.write("output/data/accident_analysis_summary.csv", summary_df)
println(summary_df)

println("\nGenerating plots")

set_theme!(Theme(
    fontsize=16,
    Axis=(
        xgridvisible=false,
        ygridvisible=true,
        ygridcolor=(:gray, 0.15),
    ),
))

fig1 = Figure(size=(900, 550), backgroundcolor=:white)
ax1 = Axis(fig1[1, 1],
    title="Średnia liczba wypadków vs liczba samochodów",
    xlabel="Liczba samochodów",
    ylabel="Średnia liczba wypadków (na 100 symulacji)",
    xticks=CAR_RANGE,
)

barplot!(ax1, summary_df.num_cars, summary_df.avg_accidents,
    color=summary_df.avg_accidents,
    colormap=:inferno,
    strokewidth=1,
    strokecolor=:gray30,
)

save("output/plots/accident_avg_plot.png", fig1, px_per_unit=2)

fig2 = Figure(size=(900, 550), backgroundcolor=:white)
ax2 = Axis(fig2[1, 1],
    title="Prawdopodobieństwo wystąpienia wypadku",
    xlabel="Liczba samochodów",
    ylabel="Prawdopodobieństwo wypadku (0-1)",
    xticks=CAR_RANGE,
    yticks=0.0:0.1:1.0,
)

ylims!(ax2, 0, 1.05)

barplot!(ax2, summary_df.num_cars, summary_df.accident_probability,
    color=summary_df.accident_probability,
    colormap=Reverse(:RdYlGn),
    strokewidth=1,
    strokecolor=:gray30,
)

for (x, p) in zip(summary_df.num_cars, summary_df.accident_probability)
    text!(ax2, x, p + 0.02; text="$(Int(round(p*100)))%",
        align=(:center, :bottom), fontsize=13, color=:gray20)
end

save("output/plots/accident_probability_plot.png", fig2, px_per_unit=2)

