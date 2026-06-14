module Training

using Agents
using BSON: @save
using CSV
using DataFrames
using Random
using ..RLWrapper
using ..TrafficEnv

export train_model, ACTIONS

const accel_options = [-2.0, 0.0, 2.0]
const steer_options = [-0.5, 0.0, 0.5]
const ACTIONS = [(a, s) for a in accel_options, s in steer_options]

function eps_greedy(Q::Dict, state::Tuple, eps::Float64)
    if rand() < eps || !haskey(Q, state)
        return rand(1:length(ACTIONS))
    else
        return argmax(Q[state])
    end
end

function train_model(; epochs=500000, max_steps=500, num_cars=4)
    println("Training with Tabular Q-Learning")

    Q = Dict{Tuple,Vector{Float64}}()

    alpha = 0.1
    gamma = 0.99
    eps_start = 1.0
    eps_end = 0.0

    epoch_rewards = Float64[]

    for epoch in 1:epochs
        c_num_cars = rand(1:num_cars)
        abm = initialize_model(num_cars=c_num_cars)

        decay_steps = epochs * 0.8
        eps = eps_start - (eps_start - eps_end) * min(1.0, epoch / decay_steps)

        total_epoch_reward = 0.0

        states = Dict{Int,Tuple}()
        actions_idx = Dict{Int,Int}()

        for step in 1:max_steps
            empty!(states)
            empty!(actions_idx)

            for agent in allagents(abm)
                if !agent.done && !agent.crashed
                    s = get_discrete_state(agent, abm)
                    states[agent.id] = s

                    if !haskey(Q, s)
                        Q[s] = zeros(Float64, length(ACTIONS))
                    end

                    a_idx = eps_greedy(Q, s, eps)
                    actions_idx[agent.id] = a_idx
                    agent.action = ACTIONS[a_idx]
                end
            end

            if isempty(states)
                break
            end

            env_step!(abm)

            for agent in allagents(abm)
                if haskey(states, agent.id)
                    s = states[agent.id]
                    a_idx = actions_idx[agent.id]

                    r = get_reward(agent)

                    total_epoch_reward += r

                    if agent.done || agent.crashed
                        Q[s][a_idx] = Q[s][a_idx] + alpha * (r - Q[s][a_idx])
                    else
                        sp = get_discrete_state(agent, abm)
                        if !haskey(Q, sp)
                            Q[sp] = zeros(Float64, length(ACTIONS))
                        end
                        max_next_Q = maximum(Q[sp])
                        Q[s][a_idx] = Q[s][a_idx] + alpha * (r + gamma * max_next_Q - Q[s][a_idx])
                    end
                end
            end
        end

        avg_car_reward = total_epoch_reward / c_num_cars
        push!(epoch_rewards, avg_car_reward)

        if epoch % 50000 == 0 || epoch == 1
            println("Epoch: $epoch, Epsilon: $(round(eps, digits=2)), Avg Reward: $(round(avg_car_reward, digits=2)), Q-Table size: $(length(Q))")
        end

        if epoch == 1 || epoch % 50000 == 0
            @save "model_checkpoint_epoch_$(epoch).bson" Q
        end
    end

    @save "model_checkpoint.bson" Q

    df = DataFrame(Epoch=1:epochs, Suma_nagrod=epoch_rewards)
    CSV.write("output/data/training_data.csv", df)

    return Q
end

end
