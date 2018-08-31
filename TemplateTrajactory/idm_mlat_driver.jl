mutable struct IDMMLATDriver <: DriverModel{LatLonAccelDirection}
    Δt::Float64
    mlon::LaneFollowingDriver
    mlat::LateralDriverModel
    longAcc::Float64
    latAcc::Float64
    turning_direction::Int
    a_lat_max::Float64
    a_cp_max::Float64
    cp_horizon::Float64
    lane_change_action::Float64
    v_des::Float64
    laneNum_desire::Int
    v_max::Float64
    v_min::Float64

    function IDMMLATDriver(
        Δt::Float64;
        mlon::LaneFollowingDriver=IDMDriver(),
        mlat::LateralDriverModel=ProportionalLaneTracker(kp=3.0,kd=2.0),
        longAcc::Float64 = 0.0,
        latAcc::Float64 = 0.0,
        turning_direction::Int = 1,
        a_lat_max::Float64 = 0.3*9.8,
        a_cp_max::Float64 = 0.3*9.8,
        cp_horizon::Float64 = 25.0,
        lane_change_action::Int = 0,
        v_des::Float64 = DESIRE_SPEED,
        laneNum_desire::Int=1,
        lane_changing::Bool=false,
        v_max::Float64=MAXIMUM_SPEED,
        v_min::Float64=MINIMUM_SPEED,
        )

        retval = new()
        retval.Δt = Δt
        retval.mlon = mlon
        retval.mlat = mlat
        retval.longAcc = longAcc
        retval.latAcc = latAcc
        retval.turning_direction = turning_direction
        retval.a_lat_max = a_lat_max
        retval.a_cp_max = a_cp_max
        retval.cp_horizon = cp_horizon
        retval.lane_change_action = lane_change_action
        retval.v_des = v_des
        retval.laneNum_desire = laneNum_desire
        retval.v_max = v_max
        retval.v_min = v_min
        retval
    end
end

function IDMMLATDriverInit(Δt::Float64=TIMESTEP,accmax::Float64=ACCELERATION_MAX,a_lat_max::Float64=ACC_LAT_MAX,a_cp_max::Float64=ACC_LAT_MAX,cp_horizon=30.0,
                            turning_direction::Int=1,s_min::Float64=S_MIN)
    mlon = IDMDriver(v_des=DESIRE_SPEED,a_max=accmax,d_max=accmax,d_cmf=accmax/2,s_min=s_min)
    # mlat = ProportionalLaneTracker()
    mlat = SmoothLaneTracker()
    return IDMMLATDriver(Δt,mlon=mlon,mlat=mlat,a_lat_max=a_lat_max,a_cp_max=a_cp_max,cp_horizon=cp_horizon,turning_direction=turning_direction)
end

AutomotiveDrivingModels.get_name(::IDMMLATDriver) = "IDMMLATDriver"

function track_longitudinal!(model::LaneFollowingDriver, scene::Union{Scene,Frame{Entity{VehicleState, BicycleModel, Int}}}, roadway::Roadway, vehicle_index::Int, fore::NeighborLongitudinalResult)
    v_ego = scene[vehicle_index].state.v
    if fore.ind != 0
        headway, v_oth = fore.Δs, scene[fore.ind].state.v
    else
        headway, v_oth = NaN, NaN
    end
    return AutoUrban.track_longitudinal!(model, v_ego, v_oth, headway)
end

function excute_action!(model::IDMMLATDriver,v_des::Float64,direction::Int,scene::Union{Scene,Frame{Entity{VehicleState, BicycleModel, Int}}}, roadway::Roadway, ego_index::Int)
    veh = scene[ego_index]
    laneNum = veh.state.posF.roadind.tag.lane
    model.laneNum_desire=laneNum+direction
    AutoUrban.set_desired_speed!(model.mlon, v_des)
    model.v_des = v_des
end

function AutomotiveDrivingModels.observe!(model::IDMMLATDriver, scene::Union{Scene,Frame{Entity{VehicleState, BicycleModel, Int}}}, roadway::Roadway, egoid::Int)
   
    vehicle_index = findfirst(scene, egoid)
    veh = scene[vehicle_index]
    x = veh.state.posG.x
    y = veh.state.posG.y
    θ = veh.state.posG.θ
    v = veh.state.v
    roadind=veh.state.posF.roadind
    laneTag=roadind.tag
    Δt = model.Δt
    laneNum=roadind.tag.lane
    frenet=Frenet(roadind, roadway[roadind].s,veh.state.posF.t,veh.state.posF.ϕ)

    laneoffset = 0.0
    lateral_speed = v*sin(veh.state.posF.ϕ)
    longitudinal_speed = v*cos(veh.state.posF.ϕ)
    
    if model.laneNum_desire==laneNum #same lane
        #println("same lane")
        laneoffset = frenet.t
        fore = get_neighbor_fore_along_lane(scene, vehicle_index, roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront())
    elseif model.laneNum_desire>laneNum #turn to left lane
        #println("left lane")
        laneoffset = frenet.t - LANE_WIDTH
        fore = NeighborLongitudinalResult(0, 250.0)
        # fore = get_neighbor_fore_along_left_lane(scene, vehicle_index, roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront())
        #=
        fore_same = get_neighbor_fore_along_lane(scene, vehicle_index, roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront())
        fore_left = get_neighbor_fore_along_left_lane(scene, vehicle_index, roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront())
        if fore_same.Δs < fore_left.Δs
            fore = fore_same
        else
            fore = fore_left
        end
        =#
    else
        #println("right lane")
        laneoffset = frenet.t - (-LANE_WIDTH)
        fore = NeighborLongitudinalResult(0, 250.0)
        # fore = get_neighbor_fore_along_right_lane(scene, vehicle_index, roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront())
        #=
        fore_same = get_neighbor_fore_along_lane(scene, vehicle_index, roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront())
        fore_right = get_neighbor_fore_along_right_lane(scene, vehicle_index, roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront())
        if fore_same.Δs < fore_right.Δs
            fore = fore_same
        else
            fore = fore_right
        end
        =#
    end

    
    # AutomotiveDrivingModels.track_lateral!(model.mlat, laneoffset, lateral_speed)
    track_lateral!(model.mlat, laneoffset, lateral_speed, longitudinal_speed)
    #AutomotiveDrivingModels.track_longitudinal!(model.mlon, scene, roadway, vehicle_index, fore.ind)
    roadind = scene[vehicle_index].state.posF.roadind
    max_k,distance = get_max_curvature(roadind, roadway, model.cp_horizon, direction = model.turning_direction)
    v_max = sqrt(model.a_cp_max/abs(max_k))
    #v_max = 5.0
    model.mlon.v_des = min(model.v_des,v_max)
    track_longitudinal!(model.mlon, scene, roadway, vehicle_index, fore)
    
    model.latAcc = clamp(rand(model.mlat),-model.a_lat_max,model.a_lat_max)
    model.longAcc = rand(model.mlon).a
    #println("a: ",model.longAcc)
    if longitudinal_speed + model.longAcc * model.Δt > model.v_max
        #println("too quick")
        model.longAcc = 0.0
    elseif longitudinal_speed + model.longAcc * model.Δt < model.v_min
        #println("too slow")
        model.longAcc = 0.0
    end
    model
end

function Base.rand(model::IDMMLATDriver)
    LatLonAccelDirection(model.latAcc, model.longAcc, model.turning_direction)
end

Distributions.pdf(model::IDMMLATDriver, a::LatLonAccelDirection) = pdf(model.mlat, a.a_lat) * pdf(model.mlon, a.a_lon)
Distributions.logpdf(model::IDMMLATDriver, a::LatLonAccelDirection) = logpdf(model.mlat, a.a_lat) * logpdf(model.mlon, a.a_lon)
