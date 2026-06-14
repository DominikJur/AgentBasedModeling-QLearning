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

const NUM_SIMS = 1000
const MAX_STEPS = 500
const CAR_RANGE = 1:12

println("Warming up JIT")
let abm = initialize_model(num_cars=4)
    for step in 1:10
        for agent in allagents(abm)
            if !agent.done && !agent.crashed
                obs = get_discrete_state(agent, abm)
                a = policy(obs)
                agent.action = (Float64(a[1]), Float64(a[2]))
            end
        end
        env_step!(abm)
    end
end

println("Running timing analysis")

rows = DataFrame(
    num_cars=Int[],
    sim_id=Int[],
    time_s=Float64[]
)

for num_cars in CAR_RANGE
    for sim in 1:NUM_SIMS
        abm = initialize_model(num_cars=num_cars)

        t = @elapsed begin
            for step in 1:MAX_STEPS
                for agent in allagents(abm)
                    if !agent.done && !agent.crashed
                        obs = get_discrete_state(agent, abm)
                        a = policy(obs)
                        agent.action = (Float64(a[1]), Float64(a[2]))
                    end
                end
                env_step!(abm)
            end
        end

        push!(rows, (num_cars=num_cars, sim_id=sim, time_s=t))
    end
    avg = mean(rows.time_s[rows.num_cars.==num_cars])
    println("  $(num_cars) car(s) — avg $(round(avg * 1000, digits=1)) ms")
end

CSV.write("output/data/timing_analysis_raw.csv", rows)

summary_df = combine(groupby(rows, :num_cars),
    :time_s => mean => :avg_time_s,
    :time_s => std => :std_time_s,
    :time_s => minimum => :min_time_s,
    :time_s => maximum => :max_time_s,
)
summary_df.avg_time_ms = summary_df.avg_time_s .* 1000
summary_df.std_time_ms = summary_df.std_time_s .* 1000

CSV.write("output/data/timing_analysis_summary.csv", summary_df)
println(summary_df)

println("\nGenerating plot")

set_theme!(Theme(
    fontsize=16,
    Axis=(
        xgridvisible=false,
        ygridvisible=true,
        ygridcolor=(:gray, 0.15),
    ),
))

fig = Figure(size=(900, 550), backgroundcolor=:white)
ax = Axis(fig[1, 1],
    title="Sredni czas symulacji vs liczba samochodow ($(MAX_STEPS) krokow, $(NUM_SIMS) powtorzen)",
    xlabel="Liczba samochodow",
    ylabel="Sredni czas symulacji [ms]",
    xticks=CAR_RANGE,
)

barplot!(ax, summary_df.num_cars, summary_df.avg_time_ms,
    color=summary_df.avg_time_ms,
    colormap=:viridis,
    strokewidth=1,
    strokecolor=:gray30,
)

err_upper = summary_df.std_time_ms
err_lower = min.(summary_df.std_time_ms, summary_df.avg_time_ms)
errorbars!(ax, summary_df.num_cars, summary_df.avg_time_ms,
    err_lower, err_upper,
    color=:gray40, linewidth=1.5, whiskerwidth=6)

for (x, v) in zip(summary_df.num_cars, summary_df.avg_time_ms)
    text!(ax, x, v + maximum(summary_df.std_time_ms) * 0.3;
        text="$(round(v, digits=1))",
        align=(:center, :bottom), fontsize=12, color=:gray20)
end

save("output/plots/timing_plot.png", fig, px_per_unit=2)
