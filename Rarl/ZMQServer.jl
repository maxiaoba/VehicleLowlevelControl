using DeepRL
using JSON
using ZMQ

include("../Ad05RTheta2RSteer2FR/preload.jl")
include("../CommonFiles/ZMQ.jl")

mutable struct JustEgoEnv <: AbstractEnvironment

    scene::Union{Scene,Frame{Entity{VehicleState, BicycleModel, Int}}}
    models::Dict{Int, DriverModel}
    roadway::Roadway

    function JustEgoEnv()
        retval = new()

        scene,models,roadway = initialize_env()

        retval.scene = scene
        retval.models = models
        retval.roadway = roadway
        return retval
    end
end


function Base.reset(env::JustEgoEnv)
    rand_ego!(env.scene,env.models,env.roadway)
    state = get_observation(env.scene,env.models,env.roadway)
    return state
end

function step!(env::JustEgoEnv, action)
    egoaction = Egoaction(action[1],action[2],action[3],action[4])
    done = simulate_action!(egoaction,env.scene,env.models,env.roadway)
    obs = get_observation(env.scene,env.models,env.roadway)
    info = nothing
    reward = reward_fn(egoaction,env.scene,env.models,env.roadway)
    return obs, reward, done, info
end

function render(env::JustEgoEnv)
    render(env.scene, env.roadway, env.models)
end

# This is what needs to be called in Julia
function run_env_server(;ip="127.0.0.1", port=9412)
    env = JustEgoEnv()
    conn = ZMQTransport(ip, port, ZMQ.REP, true)
    Logging.debug("running server...")
    while true
        msg = JSON.parse(recvreq(conn))
        Logging.info("received request: ", msg)
        respmsg = process!(env, msg)
        sendresp(conn, respmsg)
    end
    close(conn)
end
