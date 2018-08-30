mutable struct Egoaction
    acc::Float64
    steer::Float64
    dacc::Float64
    dsteer::Float64
end

function check_out_of_range(scene,roadway,ego_index)
    out_of_range=false
    veh=scene[ego_index]
    lane = roadway[veh.state.posF.roadind.tag]
    if n_lanes_right(lane, roadway) ==0 && veh.state.posF.t<-LANE_WIDTH/2
        out_of_range=true
        elseif n_lanes_left(lane, roadway) ==0 && veh.state.posF.t>LANE_WIDTH/2
        out_of_range=true
    end
    #println("out_of_range : ",out_of_range)
    θ = mod2pi(scene[1].state.posG.θ)
    if θ>pi
        θ = θ-2*pi
    end

    if abs.(θ) > THETA_MAX
        out_of_range = true
    end
    return out_of_range
end

function rand_ego!(scene=Frame(Entity{VehicleState, BicycleModel, Int},1),
                    models=Dict{Int, DriverModel}(),
                    roadway=gen_straight_roadway(LANE_NUM, 5000.0, lane_width=LANE_WIDTH);)
    
    empty!(scene)
    for key in keys(models)
        delete!(models, key)
    end

    id = 1
    #generate ego car
    ego_x=EGO_START_X
    ego_y=0.0+(2*rand()-1.0)*Y_NOISE
    ego_v=0.0+rand()*V_NOISE
    ego_theta=0.0+(2*rand()-1.0)*THETA_NOISE
    pos=VecSE2(ego_x,ego_y,ego_theta)
    push!(scene,Entity(VehicleState(pos, roadway, ego_v), 
                            BicycleModel(VehicleDef(AgentClass.CAR, CAR_LENGTH, CAR_WIDTH),CAR_A,CAR_B),id))
    models[id] = AccCurvatureStateDriver(TIMESTEP,0.0,0.0,TOTALSTEP)
    return scene,models,roadway
end

function initialize_env()
    scene=Frame(Entity{VehicleState, BicycleModel, Int},1)
    models=Dict{Int, DriverModel}()
    roadway=gen_straight_roadway(LANE_NUM, 5000.0, lane_width=LANE_WIDTH)
    return scene,models,roadway
end

function pre_simulate_action!(action::Egoaction,scene::Union{Scene,Frame{Entity{VehicleState, BicycleModel, Int}}},models::Dict{Int, DriverModel},roadway::Roadway)
    acc = clamp(action.acc,-1.0,1.0)*ACCELERATION_MAX
    steer = clamp(action.steer,-1.0,1.0)*STEER_MAX
    dacc = clamp(action.dacc,-1.0,1.0)*DACC_MAX
    dsteer = clamp(action.dsteer,-1.0,1.0)*DSTEER_MAX

    acc += dacc
    steer += dsteer

    if scene[1].state.v + acc*TIMESTEP <= MINIMUM_SPEED
        acc = 0.0
    end
    if scene[1].state.v + acc*TIMESTEP >= MAXIMUM_SPEED
        acc = 0.0
    end
    
    models[1].acc = acc
    models[1].u = steer
end

function simulate_action!(action::Egoaction,scene::Union{Scene,Frame{Entity{VehicleState, BicycleModel, Int}}},models::Dict{Int, DriverModel},roadway::Roadway)
    collision=false
    out_of_range=false
    timeout = false
    pre_simulate_action!(action,scene,models,roadway)
    actions = get_actions!(Array{Any}(length(scene)), scene, roadway, models)
    tick!(scene, roadway, actions, TIMESTEP)
    collisionResult=get_first_collision(scene, 1)
    collision=collisionResult.is_colliding
    out_of_road=check_out_of_range(scene,roadway,1)
    if scene[1].state.posG.x < (EGO_START_X - CAR_LENGTH )
         out_of_road=true
    end
    if mod2pi(scene[1].state.posG.θ) > pi/2.0 && mod2pi(scene[1].state.posG.θ) < pi*3.0/2.0
        collision=true
    end
    models[1].t -= 1
    if models[1].t == 0
        timeout=true
    end
    return collision || out_of_road || timeout
end

function get_observation(scene::Union{Scene,Frame{Entity{VehicleState, BicycleModel, Int}}},models::Dict{Int, DriverModel},roadway::Roadway)
    state = zeros(4)
    #generate ego state
    Py = scene[1].state.posG.y
    V = scene[1].state.v
    θ = mod2pi(scene[1].state.posG.θ)
    if θ>pi
        θ = θ-2*pi
    end

    state[1] = Py
    state[2] = V
    state[3] = θ
    state[4] = models[1].k
    #y v theta k
    return state
end
############################



