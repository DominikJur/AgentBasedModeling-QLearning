module TrafficEnv

using Agents
using LinearAlgebra

export Car, TrafficModel, lidar_rays, initialize_model, agent_step!, env_step!

@agent struct Car(ContinuousAgent{2, Float64})
    heading::Float64
    speed::Float64
    goal::Symbol
    color::Symbol
    crashed::Bool
    done::Bool
    wrong_lane::Bool
    action::NTuple{2, Float64}
    progress::Float64
end

function lidar_rays(agent::Car, model::StandardABM; max_dist=30.0)
    # Angles: 0, -15, -30, 15, 30 degrees
    angles = [0.0, -pi/6, pi/6]
    rays = fill((max_dist, :none), length(angles))
    
    for (i, angle) in enumerate(angles)
        ray_heading = agent.heading + angle
        ray_dir = (cos(ray_heading), sin(ray_heading))
        
        best_dist = max_dist
        best_type = :none
        
        step_size = 0.5
        for d in step_size:step_size:max_dist
            px = agent.pos[1] + ray_dir[1] * d
            py = agent.pos[2] + ray_dir[2] * d
            
            in_horizontal = 32.0 <= py <= 68.0
            in_vertical = 32.0 <= px <= 68.0
            
            if (!in_horizontal && !in_vertical)
                best_dist = d
                best_type = :wall
                break
            end
        end
        
        for other in nearby_agents(agent, model, max_dist)
            dx = other.pos[1] - agent.pos[1]
            dy = other.pos[2] - agent.pos[2]
            dist = sqrt(dx^2 + dy^2)
            
            if dist > best_dist
                continue
            end
            
            dot_prod = (dx * ray_dir[1] + dy * ray_dir[2]) / dist
            if dot_prod > 0.95
                best_dist = dist
                best_type = :car
            end
        end
        
        rays[i] = (best_dist, best_type)
    end
    
    return rays
end

function agent_step!(agent::Car, model::StandardABM)
    if agent.done || agent.crashed
        agent.action = (0.0, 0.0)
        agent.speed = 0.0
        return
    end
    
    dt = 0.1
    accel, steer = agent.action
    
    # Update heading and speed
    agent.heading += steer * dt
    agent.speed = clamp(agent.speed + accel * dt, -2.0, 10.0)
    
    agent.vel = (cos(agent.heading) * agent.speed, sin(agent.heading) * agent.speed)
    move_agent!(agent, model, dt)
    
    x, y = agent.pos
    
    in_horizontal_road = (32.0 <= y <= 68.0)
    in_vertical_road = (32.0 <= x <= 68.0)
    
    if !in_horizontal_road && !in_vertical_road
        agent.crashed = true
        agent.done = true
    end
    
    agent.wrong_lane = false
    if agent.goal == :E && in_vertical_road && !in_horizontal_road
        agent.wrong_lane = true
    elseif agent.goal == :W && in_vertical_road && !in_horizontal_road
        agent.wrong_lane = true
    elseif agent.goal == :N && in_horizontal_road && !in_vertical_road
        agent.wrong_lane = true
    elseif agent.goal == :S && in_horizontal_road && !in_vertical_road
        agent.wrong_lane = true
    end
    
    # Opposite direction driving (all 4 goals)
    h = mod(agent.heading, 2pi)
    if agent.goal == :E && (h > pi/2 && h < 3pi/2)
        # Heading west when goal is east
        if in_horizontal_road && y > 50.0
            agent.wrong_lane = true
        end
    elseif agent.goal == :W && (h < pi/2 || h > 3pi/2)
        # Heading east when goal is west
        if in_horizontal_road && y < 50.0
            agent.wrong_lane = true
        end
    elseif agent.goal == :N && (h > pi)
        # Heading south when goal is north
        if in_vertical_road && x < 50.0
            agent.wrong_lane = true
        end
    elseif agent.goal == :S && (h > 0 && h < pi)
        # Heading north when goal is south
        if in_vertical_road && x > 50.0
            agent.wrong_lane = true
        end
    end
    
    # Goal reached
    if agent.goal == :E && x > 95.0
        agent.done = true
    elseif agent.goal == :W && x < 5.0
        agent.done = true
    elseif agent.goal == :N && y > 95.0
        agent.done = true
    elseif agent.goal == :S && y < 5.0
        agent.done = true
    end
    
    # Calculate progress (distance moved towards goal)
    agent.progress = 0.0
    if agent.goal == :E
        agent.progress = agent.vel[1] * dt
    elseif agent.goal == :W
        agent.progress = -agent.vel[1] * dt
    elseif agent.goal == :N
        agent.progress = agent.vel[2] * dt
    elseif agent.goal == :S
        agent.progress = -agent.vel[2] * dt
    end
    
    for neighbor in nearby_agents(agent, model, 2.5)
        dist = sqrt((agent.pos[1] - neighbor.pos[1])^2 + (agent.pos[2] - neighbor.pos[2])^2)
        if dist <= 2.5 && !neighbor.crashed
            agent.crashed = true
            neighbor.crashed = true
            agent.done = true
            neighbor.done = true
        end
    end
end

function env_step!(model::StandardABM)
    step!(model, 1)
end

function initialize_model(; num_cars=4)
    space = ContinuousSpace((100.0, 100.0); periodic=false)
    model = StandardABM(Car, space; agent_step! = agent_step!)
    
    # Helper to add cars
    function spawn_car(pos, heading, goal, color)
        add_agent!(pos, model, (0.0, 0.0), heading, 2.0, goal, color, false, false, false, (0.0, 0.0), 0.0)
    end
    
    min_spacing = 6.0
    
    # Count cars per side (round-robin assignment)
    n_per_side = zeros(Int, 4)
    for i in 1:num_cars
        n_per_side[((i-1) % 4) + 1] += 1
    end
    
    # Side definitions: (along_start, along_end, cross_nominal, cross_min, cross_max, heading, goal, color, along_is_x)
    side_defs = [
        (3.0,  28.0, 41.0, 34.0, 48.0,  0.0,    :E, :blue, true),
        (72.0, 97.0, 59.0, 52.0, 66.0,  pi,     :W, :blue, true),
        (3.0,  28.0, 59.0, 52.0, 66.0,  pi/2,   :N, :red,  false),
        (72.0, 97.0, 41.0, 34.0, 48.0, -pi/2,   :S, :red,  false),
    ]
    
    for (side_idx, (a_start, a_end, c_nom, c_min, c_max, heading, goal, color, along_is_x)) in enumerate(side_defs)
        n = n_per_side[side_idx]
        n == 0 && continue
        
        # Evenly space base positions along the approach road
        base_positions = n == 1 ? [a_start + 2.0] : range(a_start, a_end, length=n)
        
        # Generate perturbed positions with minimum spacing guarantee
        along_final = Float64[]
        for base in base_positions
            placed = false
            for _ in 1:20
                a = clamp(base + randn() * 2.0, a_start, a_end)
                if all(abs(a - other) >= min_spacing for other in along_final)
                    push!(along_final, a)
                    placed = true
                    break
                end
            end
            !placed && push!(along_final, base)  # fallback: unperturbed
        end
        
        for a in along_final
            c = clamp(c_nom + randn() * 2.0, c_min, c_max)
            h = heading + randn() * 0.1
            if along_is_x
                spawn_car((a, c), h, goal, color)
            else
                spawn_car((c, a), h, goal, color)
            end
        end
    end
    
    return model
end

end
