using Pkg
Pkg.activate(".")

include("../src/Environment.jl")
include("../src/RLWrapper.jl")

using Agents
using .TrafficEnv
using .RLWrapper
using BSON: @load
using CSV
using DataFrames
using CairoMakie
using Statistics

println("Loading trained model")
@load "models/model_checkpoint.bson" Q

accel_options = [-2.0, 0.0, 2.0]
steer_options = [-0.5, 0.0, 0.5]
ACTIONS = [(a, s) for a in accel_options, s in steer_options]

policy = (obs) -> begin
    if haskey(Q, obs)
        return ACTIONS[argmax(Q[obs])]
    else
        return (0.0, 0.0)
    end
end

const NUM_SIMS = 100
const MAX_STEPS = 500
const CAR_RANGE = 1:12


all_rows = DataFrame(
    num_cars=Int[],
    sim_id=Int[],
    num_accidents=Int[],
    any_accident=Bool[]
)

for num_cars in CAR_RANGE
    for sim in 1:NUM_SIMS
        abm = initialize_model(num_cars=num_cars)

        for step in 1:MAX_STEPS
            for agent in allagents(abm)
                if !agent.done && !agent.crashed
                    obs = get_discrete_state(agent, abm)
                    a = policy(obs)
                    agent.action = (Float64(a[1]), Float64(a[2]))
                end
            end
            env_step!(abm)

            if all(a -> a.done, allagents(abm))
                break
            end
        end

        crashed_count = count(a -> a.crashed, allagents(abm))

        push!(all_rows, (
            num_cars=num_cars,
            sim_id=sim,
            num_accidents=crashed_count,
            any_accident=crashed_count > 0
        ))
    end
    println("$(num_cars) car(s) done")
end

CSV.write("output/data/accident_analysis_raw.csv", all_rows)

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

