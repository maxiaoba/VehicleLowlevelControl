function log_speed(speed)
    x = clamp(speed,MINIMUM_SPEED,DESIRE_SPEED)
    x = (speed/MAXIMUM_SPEED) * 100.0 + 0.99
    x = max(0,x)
    base = MAXIMUM_SPEED/2.0
    reward = max(0.0, log(base, x) / 2.0)
    reward -= 0.5
    reward *= 2
    return reward
end

function reward_fn(action::Egoaction,fail::Bool,
        scene::Union{Scene,Frame{Entity{VehicleState, BicycleModel, Int}}},models::Dict{Int, DriverModel},roadway::Roadway)
    if fail
        return -200.0
    end

    acc = action.acc*ACCELERATION_MAX
    steer = action.steer*STEER_MAX
    acc = max(-ACCELERATION_MAX,min(ACCELERATION_MAX,acc))
    steer = max(-STEER_MAX,min(STEER_MAX,steer))
    
    speedReward = log_speed(scene[1].state.v)
    Px = scene[1].state.posG.x
    Py = scene[1].state.posG.y
    V = scene[1].state.v
    θ = mod2pi(scene[1].state.posG.θ)
    goal = LANE_WIDTH
    t = models[1].t
    
    goalReward = 0.0
    if abs(Py-goal) < 0.05
        goalReward = 3.0
    elseif abs(Py-goal) < LANE_WIDTH
        goalReward = 1.0-abs(Py-goal)/LANE_WIDTH
    else
        goalReward = 1.0-abs(Py-goal)/LANE_WIDTH
    end
    
    accReward = -abs(acc)/ACCELERATION_MAX
    steerReward = -abs(steer)/STEER_MAX
    
    thetaReward = 0.0
    if θ>pi
        θ = θ-2*pi
    end
    thetaReward = 1.0-abs.(θ)/(pi/4)
    if abs(θ) > pi/4
        thetaReward = -1.0
    # else
    #     if (goal > Py) && (θ < -0.1)
    #         thetaReward = -1.0
    #     elseif (goal < Py) && (θ > 0.1)
    #         thetaReward = -1.0
    #     else
    #         thetaReward = 0.5-abs(θ)/(pi/4)
    #     end
    end
    
    return 0.5*speedReward+goalReward+0.1*accReward+2.0*steerReward+2.0*thetaReward
end
