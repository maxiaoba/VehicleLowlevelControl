
export AccCurvatureStateDriver

mutable struct AccCurvatureStateDriver <: DriverModel{AccelSteeringDirection}
    dt::Float64
    acc::Float64
    u::Float64 #commadn steer
    c::Array{Float64}
    direction::Int
    Ad::Array{Float64}
    Bd::Array{Float64}
    Cd::Array{Float64}
    steer::Float64
    k::Float64 #actual curvature
    t::Int
    expertState::VehicleState
    oldAcc::Float64
    oldSteer::Float64
end

function AccCurvatureStateDriver(dt::Float64, a::Float64, u::Float64,t::Int;
        c::Array{Float64}=zeros(2,1), direction::Int=1,
        A::Array{Float64} = ACONST,
        B::Array{Float64} = BCONST,
        C::Array{Float64} = CCONST,
        steer::Float64 = 0.0,
        k::Float64 = 0.0,
        expertState::VehicleState = VehicleState(),
        oldAcc::Float64 = 0.0,
        oldSteer::Float64 = 0.0,
        )
    M=expm([A B; 0 0 0]*dt)
    Ad=M[1:2,1:2]
    Bd=M[1:2,3]
    Cd = C
    return AccCurvatureStateDriver(dt,a,u,c,direction,Ad,Bd,Cd,steer,k,t,expertState,oldAcc,oldSteer)
end

AutomotiveDrivingModels.get_name(model::AccCurvatureStateDriver) = "AccCurvatureStateDriver"

function AutomotiveDrivingModels.observe!(model::AccCurvatureStateDriver, scene::Scene, roadway::Roadway, egoid::Int)
    model
end

function Base.rand(model::AccCurvatureStateDriver)
    L = AXLE_DISTANCE
    model.c = model.Ad*model.c + model.Bd*tan.(model.u)/L
    model.k = (model.Cd*model.c)[1]
    model.steer = atan(L*model.k)
    return AccelSteeringDirection(model.acc,model.steer,model.direction)
end
Distributions.pdf(model::AccCurvatureStateDriver, a::Float64) = a == model.acc ? Inf : 0.0
Distributions.logpdf(model::AccCurvatureStateDriver, a::Float64) = a == model.acc ? Inf : -Inf