module RLWrapper

using Agents
using ..TrafficEnv

export get_discrete_state, get_reward

function get_discrete_state(agent::Car, model::StandardABM)
    x_bin = clamp(ceil(Int, agent.pos[1] / 10.0), 1, 10)
    y_bin = clamp(ceil(Int, agent.pos[2] / 10.0), 1, 10)

    h = mod(agent.heading, 2pi)
    if h >= 7pi / 4 || h < pi / 4
        heading_bin = 1
    elseif h >= pi / 4 && h < 3pi / 4
        heading_bin = 2
    elseif h >= 3pi / 4 && h < 5pi / 4
        heading_bin = 3
    else
        heading_bin = 4
    end

    rays = lidar_rays(agent, model)

    function parse_ray(r)
        dist, typ = r
        if dist >= 10.0
            return 1
        else
            return typ == :wall ? 2 : 3
        end
    end

    r1 = parse_ray(rays[1])
    r2 = parse_ray(rays[2])
    r3 = parse_ray(rays[3])

    return (x_bin, y_bin, agent.goal, heading_bin, r1, r2, r3)
end

function get_reward(agent::Car)
    r = 0.0

    r += agent.progress * 2.0

    r -= 0.5

    if agent.crashed
        r -= 1000.0
    end
    if agent.wrong_lane
        r -= 5.0
    end

    if agent.done && !agent.crashed && !agent.wrong_lane
        r += 1000.0
    end

    return r
end

end # module
