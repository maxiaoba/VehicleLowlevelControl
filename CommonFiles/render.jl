import AutoViz.render
import AutoViz.render!

function render_mpc_drivers!(
    rendermodel::RenderModel,
    model::MultiPtsDriver,
    vehid::Int,
    car_colors::Dict{Int,Colorant}
    )
    
    add_instruction!(rendermodel, render_point_trail, (model.Pts,model.color,0.25))
    add_instruction!(rendermodel, render_point_trail, (model.Pts_smoothed,model.color2,0.25))
    if isdefined(model, :Pts_predicted)
        add_instruction!(rendermodel, render_point_trail, (model.Pts_predicted,model.color3,0.25))
    end
    # if isdefined(model, :Pts_pre)
    #     add_instruction!(rendermodel, render_point_trail, (model.Pts_pre,model.color4,0.25))
    # end
    car_colors[vehid]=model.color
    rendermodel
end

function render_multipoint_drivers!(
    rendermodel::RenderModel,
    model::MultiPtsDriver,
    vehid::Int,
    car_colors::Dict{Int,Colorant}
    )
    
    add_instruction!(rendermodel, render_point_trail, (model.Pts,model.color,0.25))
    car_colors[vehid]=model.color
    rendermodel
end

function render_urban_drivers!(
    rendermodel::RenderModel,
    model::UrbanDriver,
    vehid::Int,
    car_colors::Dict{Int,Colorant}
    )

    car_colors[vehid]=model.color
    rendermodel
end

function render(ctx::CairoContext, scene::Union{Scene,Frame{Entity{VehicleState, BicycleModel, Int}}}, roadway::Roadway,models::Dict{Int, DriverModel};
    text::Vector{String}=["Nothing"],
    #trafficlights::Vector{TrafficLight}=TrafficLight[],
    rendermodel::RenderModel=RenderModel(),
    cam::Camera=SceneFollowCamera(),
    car_colors::Dict{Int,Colorant}=Dict{Int,Colorant}()
    )

    canvas_width = floor(Int, Cairo.width(ctx))
    canvas_height = floor(Int, Cairo.height(ctx))

    clear_setup!(rendermodel)

    render!(rendermodel, roadway)
    
    #render_multipoint_drivers!(rendermodel,models[2],1,car_colors)

    
    render!(rendermodel, scene, car_colors=car_colors)
    
    textoverlay=TextOverlay(text,colorant"white",20,VecE2(10, 20),1.5,false)
    render!(rendermodel, textoverlay, scene, roadway)
    
    #render_trafficlights!(rendermodel,trafficlights)
    
    camera_set!(rendermodel, cam, scene, roadway, canvas_width, canvas_height)

    render(rendermodel, ctx, canvas_width, canvas_height)
    ctx
end

function render(scene::Union{Scene,Frame{Entity{VehicleState, BicycleModel, Int}}}, roadway::Roadway,models::Dict{Int, DriverModel};
    text::Vector{String}=["Nothing"],
    #trafficlights::Vector{TrafficLight}=TrafficLight[],
    canvas_width::Int=DEFAULT_CANVAS_WIDTH,
    canvas_height::Int=DEFAULT_CANVAS_HEIGHT,
    rendermodel::RenderModel=RenderModel(),
    cam::Camera=SceneFollowCamera(),
    car_colors::Dict{Int,Colorant}=Dict{Int,Colorant}(), # id
    )

    s, ctx = get_surface_and_context(canvas_width, canvas_height)
    render(ctx, scene, roadway,models, rendermodel=rendermodel, cam=cam, car_colors=car_colors,text=text#=,trafficlights=trafficlights=#)
    s
end