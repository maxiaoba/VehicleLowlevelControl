using DeepRL
using JSON
using ZMQ

struct ZMQTransport
    ctx::Context
    sock::Socket
    mode::Int
    bound::Bool
end
function ZMQTransport(addr::String, mode::Int, bound::Bool, ctx::Context=Context())
    sock = Socket(ctx, mode)
    if bound
        ZMQ.bind(sock, addr)
    else
        ZMQ.connect(sock, addr)
    end
    ZMQTransport(ctx, sock, mode, bound)
end
function ZMQTransport(ip, port, mode::Int, bound::Bool, ctx::Context=Context())
    ZMQTransport("tcp://$ip:$port", mode, bound, ctx)
end

"""Close connection to the socket"""
function close(conn::ZMQTransport)
    Logging.debug("closing connection...")
    close(conn.sock)
end

"""Receive request from the socket"""
function recvreq(conn::ZMQTransport)
    reqstr = unsafe_string(ZMQ.recv(conn.sock))
    Logging.debug("received request: ", reqstr)
    reqstr
end

"""Send response to the socket"""
function sendresp(conn::ZMQTransport, msgstr)
    Logging.debug("sending response: ", msgstr)
    ZMQ.send(conn.sock, JSON.json(msgstr))
end

function process!(env::AbstractEnvironment, msg::Dict{String, Any})
    if "cmd" in keys(msg)
        if msg["cmd"] == "obs_dimensions"
            respmsg = Dict("obs_dim"=>obs_dimensions(env))
        elseif msg["cmd"] == "n_actions"
            respmsg = Dict("n_actions"=>n_actions(env))
        elseif msg["cmd"] == "render"
            render(env)
        elseif msg["cmd"] == "reset"
            obs = reset(env)
            respmsg = Dict("obs"=> obs)
        elseif msg["cmd"] == "step"
            action = msg["action"]
            obs, rew, done, info = step!(env, action)
            respmsg = Dict("obs"=> obs, "rew"=> rew, "done"=>done, "info"=>info)
        else
            respmsg = Dict("error"=>"no known "+msg["cmd"]+" cmd found")
        end
    end
    respmsg
end