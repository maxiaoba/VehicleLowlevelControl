function reward_fn(action::Egoaction,fail::Bool,
        scene::Union{Scene,Frame{Entity{VehicleState, BicycleModel, Int}}},models::Dict{Int, DriverModel},roadway::Roadway)
    if fail
        return -200.0
    end

    oldacc = models[1].oldAcc
    oldsteer = models[1].oldSteer

    acc = clamp(action.acc,-1.0,1.0)*ACCELERATION_MAX
    steer = clamp(action.steer,-1.0,1.0)*STEER_MAX

    models[1].oldAcc = acc
    models[1].oldSteer = steer

    Px = scene[1].state.posG.x
    Py = scene[1].state.posG.y
    Pv = scene[1].state.v
    Pθ = mod2pi(scene[1].state.posG.θ)
    if Pθ>pi
        Pθ = Pθ-2*pi
    end

    Ex = models[1].expertState.posG.x
    Ey = models[1].expertState.posG.y
    Ev = models[1].expertState.v
    Eθ = mod2pi(models[1].expertState.posG.θ)
    if Eθ>pi
        Eθ = Eθ-2*pi
    end
    goal = LANE_WIDTH
    t = models[1].t

    goal = LANE_WIDTH
    t = models[1].t
    
    alpha = 0.05#0.02
    alpha2 = 2.0#1.0
    pos_weight = 4.0
    steer_weight = 10.0

    trace_cost = pos_weight*((Px-Ex)^2+(Py-Ey)^2)+
                (sin(Pθ)-sin(Eθ))^2+(cos(Pθ)-cos(Eθ))^2
                +(Pv-Ev)^2
    control_cost = acc^2+steer_weight*steer^2
    smooth_cost = (acc-oldacc)^2+steer_weight*(steer-oldsteer)^2
    reward = -0.1*(trace_cost+alpha*control_cost+alpha2*smooth_cost)
    
    return reward
end
