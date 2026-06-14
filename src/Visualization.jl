module Visualization

using CairoMakie
using CSV
using DataFrames
using Agents
using GeometryBasics
using ..TrafficEnv
using ..RLWrapper
using BSON: @load

export create_animation, create_timelapse
function setup_animation_observables(ax, num_cars)
    car_pos = Observable(Point2f[])
    car_rot = Observable(Float32[])
    car_colors = Observable(Symbol[])
    arrow_pts = Observable(Point2f[])
    arrow_dirs = Observable(Vec2f[])
    lidar_pts = Observable(Point2f[])
    lidar_colors = Observable(Symbol[])

    function update_observables!(abm)
        pos_list = Point2f[]
        rot_list = Float32[]
        colors = Symbol[]
        pts = Point2f[]
        dirs = Vec2f[]
        l_pts = Point2f[]
        l_cols = Symbol[]


        angles = (0.0, -pi / 6, pi / 6)

        for car in allagents(abm)
            if car.done && !car.crashed && (car.pos[1] < 0 || car.pos[1] > 100 || car.pos[2] < 0 || car.pos[2] > 100)
                continue
            end

            cx, cy = car.pos
            theta = car.heading

            push!(pos_list, Point2f(cx, cy))
            push!(rot_list, Float32(theta))

            if car.crashed
                push!(colors, :black)
            else
                push!(colors, car.color)
            end

            push!(pts, Point2f(cx, cy))
            push!(dirs, Vec2f(cos(theta) * 3.0, sin(theta) * 3.0))

            rays = lidar_rays(car, abm; max_dist=30.0)
            for (i, angle) in enumerate(angles)
                ray_len, ray_type = rays[i]
                hit = ray_len < 10.0

                draw_len = ray_len

                ray_heading = theta + angle
                ex = cx + cos(ray_heading) * draw_len
                ey = cy + sin(ray_heading) * draw_len

                push!(l_pts, Point2f(cx, cy))
                push!(l_pts, Point2f(ex, ey))

                color = :white
                if hit
                    if ray_type == :car
                        color = :red
                    elseif ray_type == :wall
                        color = :orange
                    end
                end
                push!(l_cols, color)
            end
        end

        car_pos[] = pos_list
        car_rot[] = rot_list
        car_colors[] = colors
        arrow_pts[] = pts
        arrow_dirs[] = dirs
        lidar_pts[] = l_pts
        lidar_colors[] = l_cols
    end

    return car_pos, car_rot, car_colors, arrow_pts, arrow_dirs, lidar_pts, lidar_colors, update_observables!
end

function create_timelapse(; num_cars=8, max_steps=150, model_files=["model_checkpoint.bson"], output_file="output/videos/traffic_simulation_timelapse.mp4")
    println("Creating timelapse")

    fig = Figure(size=(800, 800))
    ax = Axis(fig[1, 1], backgroundcolor=:green)
    hidespines!(ax)
    hidedecorations!(ax)

    poly!(ax, Point2f[(0, 32), (100, 32), (100, 68), (0, 68)], color=:gray)
    poly!(ax, Point2f[(32, 0), (68, 0), (68, 100), (32, 100)], color=:gray)

    car_pos, car_rot, car_colors, arrow_pts, arrow_dirs, lidar_pts, lidar_colors, update_observables! = setup_animation_observables(ax, num_cars)

    epoch_text = Observable("Initializing")
    text!(ax, epoch_text, position=(5, 95), color=:white, fontsize=30)

    abm = initialize_model(num_cars=num_cars)
    update_observables!(abm)

    linesegments!(ax, lidar_pts, color=lidar_colors, linewidth=2.0)
    scatter!(ax, car_pos, marker=:rect, rotation=car_rot, color=car_colors, markersize=Vec2f(4.0, 2.0), markerspace=:data)
    arrows2d!(ax, arrow_pts, arrow_dirs, color=:white, lengthscale=1.0)

    xlims!(ax, 0, 100)
    ylims!(ax, 0, 100)

    rm(output_file, force=true)
    stream = VideoStream(fig, framerate=75)

    for model_file in model_files
        if !isfile(model_file)
            continue
        end

        @load model_file Q
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

        epoch_str = match(r"epoch_(\d+)", model_file)
        if epoch_str !== nothing
            epoch_text[] = "Epoch $(epoch_str.captures[1])"
        else
            epoch_text[] = "Fully Trained"
        end

        abm = initialize_model(num_cars=num_cars)

        for step in 1:max_steps
            for agent in allagents(abm)
                if !agent.done && !agent.crashed
                    obs = get_discrete_state(agent, abm)
                    a = policy(obs)
                    agent.action = (Float64(a[1]), Float64(a[2]))
                end
            end
            env_step!(abm)
            update_observables!(abm)
            recordframe!(stream)

            if all(a -> a.done, allagents(abm))
                for _ in 1:10
                    recordframe!(stream)
                end
                break
            end
        end
    end

    save(output_file, stream)
end

function create_animation(; num_cars=8, max_steps=150, model_file="model_checkpoint.bson", output_file="output/videos/traffic_simulation.mp4")
    create_timelapse(num_cars=num_cars, max_steps=max_steps, model_files=[model_file], output_file=output_file)
end

end
