mutable struct SmoothLaneTracker <: LateralDriverModel
    a::Float64 # predicted acceleration
    σ::Float64 # optional stdev on top of the model, set to zero or NaN for deterministic behavior
    kp::Float64 # proportional constant for lane tracking
    kd::Float64 # derivative constant for lane tracking
    k_max::Float64 # maximum curvature
    phi_max::Float64 # maximum relative angle

    function SmoothLaneTracker(;
        σ::Float64 = NaN,
        kp::Float64 = 3.0,
        kd::Float64 = 10.0,
        k_max::Float64 = tan(STEER_MAX/4.0)/AXLE_DISTANCE,
        phi_max::Float64 = 15/180*pi,
        )

        retval = new()
        retval.a = NaN
        retval.σ = σ
        retval.kp = kp
        retval.kd = kd
        retval.k_max = k_max
        retval.phi_max = phi_max
        retval
    end
end

function track_lateral!(model::SmoothLaneTracker, laneoffset::Float64, lateral_speed::Float64, longitudinal_speed::Float64)
    v = sqrt(lateral_speed^2+longitudinal_speed^2)
    a_max = v*longitudinal_speed*model.k_max
    model.a = clamp(-laneoffset*model.kp - lateral_speed*model.kd,-a_max,a_max)
    if abs(lateral_speed/(longitudinal_speed+0.01))>tan(model.phi_max)
        if lateral_speed>0.0 && model.a>0.0
            model.a = 0.0
        elseif lateral_speed<0.0 && model.a<0.0
            model.a =0.0
        end
    end
    #println("laneoffset: ",laneoffset," lateral speed: ",lateral_speed," lateral acc: ",model.a)
    model
end

function Base.rand(model::SmoothLaneTracker)
    if isnan(model.σ) || model.σ ≤ 0.0
        model.a
    else
        rand(Normal(model.a, model.σ))
    end
end