#load libraries
using Dates
using CSV
using Statistics
using Random
using LinearAlgebra
using DataFrames
using CUDA
@show CUDA.functional(true)

using Zygote
using Flux
using Functors
using MLDatasets
using Images
using cuDNN
using Distributed
using BenchmarkTools
using ProgressMeter
using JLD2
using TOML
using Images: channelview
using Images.ImageCore
using Base.Iterators: partition
using Flux, Flux.Optimise
using Flux: onehotbatch, onecold, Flux.flatten
using Flux: Momentum
using LinearAlgebra: dot, norm
using DataFrames: DataFrame
using Serialization

ones32(dims::Integer...) = Base.ones(Float32, dims...)
zeros32(dims::Integer...) = Base.zeros(Float32, dims...)

"""Get the required datasets"""
function get_fashion_mnist_dataset(;include_class=[0,1,2,3,4,5,6,7,8,9,10], label_smoothing = 0)
    # Download the FashionMNISt dataset
    # Get all the images and labels
    raw_train_labels = FashionMNIST(split=:train).targets #extract all train labels
    train_locs_to_include=[label in include_class for label in raw_train_labels] #chose labels included in include_class
    train_labels=raw_train_labels[train_locs_to_include] #extract labels
    n_train_samples=length(train_labels) #count number of samples
    Ytrain = onehotbatch([train_labels[i] for i in 1:n_train_samples], include_class) #onehotbatch labels

    raw_train_data = Flux.unsqueeze(float(FashionMNIST(split=:train).features),3) #extract all train images
    Xtrain=raw_train_data[:,:,:,train_locs_to_include] #extract train images

    raw_test_labels = FashionMNIST(split=:test).targets #extract all test labels
    test_locs_to_include=[label in include_class for label in raw_test_labels] #chose labels included in include_class
    test_labels=raw_test_labels[test_locs_to_include] #extract labels
    n_test_samples=length(test_labels) #count number of samples
    Ytest = onehotbatch([test_labels[i] for i in 1:n_test_samples], include_class)

    raw_test_data = Flux.unsqueeze(float(FashionMNIST(split=:test).features),3) #extract all train images
    Xtest=raw_test_data[:,:,:,test_locs_to_include] #extract train images

    if label_smoothing != 0
        Ytrain = Flux.label_smoothing.(Ytrain, label_smoothing)
        Ytest = Flux.label_smoothing(Ytest, label_smoothing)
    end
    return gpu(Xtrain), gpu(Ytrain), gpu(Xtest), gpu(Ytest)
end

function select_samples(data, samples_to_use)
    """Function to select a subset of training data for minibatching"""
    dim=ndims(data)
    ind = ntuple(d -> d == dim ? samples_to_use : :, ndims(data))

    return data[ind...]
end

"""
Define Perturb layer. Perturb layer is a custom layer used in Node Perturbation training method.
It adds a bias to the input x provided to the layer and retains the same shape.
it's similar to N->N dense layer with all with w_{ii}=1 and w_{ij}(i!=j) =0.
The nonlinearity of the previous layer is also transferred to Perturb layer
"""
struct Perturb{B,F}
    bias::B
    σ::F
end

function (a::Perturb)(x::AbstractArray)
    a.σ.(x.+ a.bias)
end
@functor Perturb


struct PMGD_layer{
    Layer, #the actual layer to be trained
    Train_method, #node perturbation or weight perturbation
    X, #input to the layer, required for node perturbation training method
    Outputsize, #size of the layer output
    Theta_per, #pointer to the parameters that should be perturbed
    Theta_ac, #perturbations added to parameters being trained (theta_per)
    G, #instantenous gradient defined as d(Cost)/d(theta_per) 
    G_int, #G integrated for t_theta
    M_int, #Adams optimizer momentum or just momentum
    M2_int, #Adams optimizer RMSProp
    Norm_mag, #norm_mag defined as sqrt(Num_params)*Amplitude_per
    Num_params, #number of parameters to be perturbed
    Amplitude_per, #the amplitude of theta_per perturbation
    }
    """Define PMGD_layer. PMGD_layer Struct saves the all the parameters needed in 
    weight/node perturbation PMGD training method."""

layer::Layer
train_method::Train_method
x::X
outputsize::Outputsize
theta_per::Theta_per
theta_ac::Theta_ac
G::G
G_int::G_int
M_int::M_int
M2_int::M2_int
norm_mag::Norm_mag
num_params::Num_params
amplitude_per::Amplitude_per
end

function PMGDify(
    Layer,
    Train_method,
    norm_mag,
    rng,
    x,
    )
    """Takes a layer, training method and input to the layer x
    and returns correctly Initialised PMGD_layer struct"""
    
    # initialize layer with random but reproducible parameters
    # is provided (if rng with the same seed is provided)
    if (Layer isa Dense) || (Layer isa Conv)
        Layer.weight.=Flux.glorot_uniform(rng,size(Layer.weight)...)
        Layer.bias.=0
    end

    y=Layer(x) #calculate the output of the layer
    outputsize=size(y) #calculate the size of the output of the layer

    if length(Flux.params(Layer))==0
        Train_method = "none"
    end
    if Train_method != "none"
        num_params_base=sum(length.(Flux.params(Layer)))
    else
        num_params_base=1
    end

    # Create a placeholder for all Gradients
    G=[] 
    for param in Flux.params(Layer)
        push!(G,zeros(size(param)))
    end    
    G=gpu(G)
    G_int = deepcopy.(G)
    M_int = deepcopy.(G_int)
    M2_int = deepcopy.(G_int)

    if Train_method == "none"
        theta_per=Params([]) #no parameter to perturb
    end
    if (Train_method == "weight") || (Train_method == "backprop")
        theta_per=Flux.params(Layer) #perturb all parameters
    end
    if Train_method == "node"
        if Layer isa Dense
            #create a new dense layer which is same as input dense layer without the nonlinearity of the activation
            new_dense=Dense(Layer.weight,Layer.bias) |> gpu 
            #create a new perturb layer which will perturb the activation, and then apply the nonlinearity
            pert_layer=Perturb(zeros(Float32, size(y)[1:end-1]),Layer.σ) |> gpu
            # combine the previous 2 into single layer and perturb the activations only
            Layer=Chain(new_dense,pert_layer)
            theta_per=Flux.params(pert_layer)
        end
        if Layer isa Conv
            #create a new conv layer which same as input conv layer without the nonlinearity of the activation
            new_conv=Conv(Layer.weight, Layer.bias, stride = Layer.stride, pad = Layer.pad, dilation = Layer.dilation, groups = Layer.groups) |> gpu
            #create a new perturb layer which will perturb the activation, and then apply the nonlinearity
            pert_layer=Perturb(zeros(Float32, size(y)[1:end-1]),Layer.σ) |> gpu
            # combine the previous 2 into single layer and perturb the activations only
            Layer=Chain(new_conv,pert_layer)
            theta_per=Flux.params(pert_layer)
        end        
    end

    if Train_method != "none"
        num_params=sum(length.(theta_per))
    else
        num_params=1
    end

    @show norm_mag=Float32(norm_mag)
    @show num_params
    @show amplitude_per=Float32(norm_mag/sqrt(num_params))

    # Make a duplicate copy of the Flux Chain parameters to store the perturbation
    theta_ac = deepcopy.(theta_per).*0
    return PMGD_layer(Layer, Train_method, x, outputsize, theta_per, theta_ac, G, G_int, M_int, M2_int, norm_mag, num_params, amplitude_per)
end

function Chain_from_Struct(Struct_Chain)    
    """Create f_eval_train from a struct"""
    layer_list=[]
    for item in Struct_Chain
        push!(layer_list,item.layer)
    end
    return Chain(layer_list)
end

function RawChain_from_Struct(Struct_Chain)    
    """Create f_eval_inference from a struct which ignores perturb layers"""
    layer_list=[]
    for Struct_layer in Struct_Chain
        if Struct_layer.train_method == "node"
            Layer=Struct_layer.layer[1]
            if Layer isa Dense
                new_dense=Dense(Layer.weight,Layer.bias,Struct_layer.layer[2].σ) |> gpu
                push!(layer_list,new_dense)
            end
            if Layer isa Conv
                new_conv=Conv(Layer.weight, Layer.bias, Struct_layer.layer[2].σ, stride = Layer.stride, pad = Layer.pad, dilation = Layer.dilation, groups = Layer.groups) |> gpu
                push!(layer_list,new_conv)
            end
        else
            push!(layer_list,Struct_layer.layer)
        end
    end
    return Chain(layer_list)
end

function update_G!(PMGDlayer,C_ac,tau_theta)
    """ Compute G, the psuedo-gradient, update G in place"""
    batch_size=size(PMGDlayer.x)[end]
    num_params=PMGDlayer.num_params
    norm_mag=PMGDlayer.norm_mag

    error_signal=[]
    for i in 1:length(PMGDlayer.theta_per)
        #Error signal is calculated as Cac*theta_Ac for every sample input in the minibatch        
        #reshape theta_ac[i] to a column such that 5 x 5 matrix becomes 25 x 1 matrix
        theta_ac_1D=reshape(PMGDlayer.theta_ac[i],(:,1))
        #reshape C_ac to a row such that it is a 1 x batch_size matrix
        C_ac_1d=reshape(C_ac,(1,:))
        #multiply theta_ac[i]xC_ac and reshape to get 5 x 5 x batch_size output
        push!(error_signal,reshape(theta_ac_1D*C_ac_1d,(size(PMGDlayer.theta_per[i])...,:)))
    end

    if (PMGDlayer.train_method == "weight")
        for i in 1:length(PMGDlayer.theta_per)
            #take mean along batch_size index
            PMGDlayer.G[i].= mean(error_signal[i],dims=ndims(error_signal[i]))
        end
    end

    if (PMGDlayer.train_method == "node") && (PMGDlayer.layer[1] isa Dense)
        PMGDlayer.G[1].= error_signal[1]*transpose(PMGDlayer.x)./(batch_size)
        PMGDlayer.G[2].= mean(error_signal[1],dims=ndims(error_signal[1]))
    end
    if (PMGDlayer.train_method == "node") && (PMGDlayer.layer[1] isa Conv)
        #give info about conv filter
        cdims = DenseConvDims(size(PMGDlayer.x), size(PMGDlayer.layer[1].weight), stride = PMGDlayer.layer[1].stride, padding = PMGDlayer.layer[1].pad, dilation = PMGDlayer.layer[1].dilation, groups = PMGDlayer.layer[1].groups,)
        #calculate gradients normalized by batch_size
        PMGDlayer.G[1].= ∇conv_filter(PMGDlayer.x, error_signal[1], cdims)./(batch_size)
        PMGDlayer.G[2].= sum(error_signal[1],dims=(1,2,4))[1,1,:]./(batch_size)
    end
    # Compute G_int: integrate G discretely
    PMGDlayer.G_int[1] .+= PMGDlayer.G[1]./norm_mag./sqrt(1/num_params+1/tau_theta-1/num_params/tau_theta)
    PMGDlayer.G_int[2] .+= PMGDlayer.G[2]./norm_mag./sqrt(1/num_params+1/tau_theta-1/num_params/tau_theta)
end

function G_true_from_Struct(x,y_target,f_cost,Struct_Chain)
    """Makes a list of the true gradient G_true=[G[w_matrix layer 1],G[b_matrix layer 1],G[w_matrix layer 2],G[b_matrix layer 2]...] required for angle calculations"""
    f_eval=RawChain_from_Struct(Struct_Chain)
    theta=Flux.params(f_eval)
    G_true = deepcopy.(theta)
    gs = gradient(() -> mean(f_cost(f_eval(x), y_target)), Flux.params(f_eval))
    for i in 1:length(theta)
        G_true[i] .= gs[theta[i]]
    end
    return G_true
end

function _angle_between(a, b)
    y::Float32 = dot(a,b)/(norm(a)*norm(b))
    return rad2deg(acos(clamp(y, -1.0, 1.0)))
end

function _ratio_between(a, b)
    y::Float32 = norm(a)/norm(b)
    return y
end

function _distance_between(a, b)
    y::Float32 = norm(a.-b)/norm(b)
    return y
end

function compute_angle_from_Struct(x,y_target,f_cost,Struct_Chain,iteration)
    """Computes angle between G_true and G_int and
    returns both the angle and componentwise convergence 
    G_true[1] dot G_int[1]
    G_true[2] dot G_int[2]..."""
    G_int=[item./iteration for PMGDlayer in Struct_Chain for item in PMGDlayer.G_int]
    G_true = G_true_from_Struct(x,y_target,f_cost,Struct_Chain)

    angle_list=[]
    ratio_list=[]
    distance_list=[]

    push!(angle_list,_angle_between(G_int,G_true))
    push!(ratio_list,_ratio_between(G_int,G_true))
    push!(distance_list,_distance_between(G_int,G_true))

    for index in range(1,length(G_true))
        push!(angle_list,_angle_between(G_int[(index-1)%length(G_int)+1],G_true[(index-1)%length(G_true)+1]))
        push!(ratio_list,_ratio_between(G_int[(index-1)%length(G_int)+1],G_true[(index-1)%length(G_true)+1]))
        push!(distance_list,_distance_between(G_int[(index-1)%length(G_int)+1],G_true[(index-1)%length(G_true)+1]))
    end

    return angle_list, ratio_list, distance_list
end

function update_theta!(PMGDlayer, eta, tau_theta, optimizer, gamma, beta1, beta2)
    if optimizer=="vanilla"
        """ Update parameteres in place, needs train_method"""
        if (PMGDlayer.train_method == "weight") || (PMGDlayer.train_method == "backprop")
            PMGDlayer.layer.weight .-= eta*PMGDlayer.G_int[1]/tau_theta #update weights
            PMGDlayer.layer.bias   .-= eta*PMGDlayer.G_int[2]/tau_theta #update biases
        end
        if (PMGDlayer.train_method == "node")
            PMGDlayer.layer[1].weight .-= eta*PMGDlayer.G_int[1]/tau_theta #update weigths
            PMGDlayer.layer[1].bias   .-= eta*PMGDlayer.G_int[2]/tau_theta #update biases 
        end
    end
    if optimizer=="momentum"
        PMGDlayer.M_int[1] .= gamma*PMGDlayer.M_int[1] .+ eta*PMGDlayer.G_int[1]/tau_theta
        PMGDlayer.M_int[2] .= gamma*PMGDlayer.M_int[2] .+ eta*PMGDlayer.G_int[2]/tau_theta
        """ Update parameteres in place, needs train_method"""
        if (PMGDlayer.train_method == "weight") || (PMGDlayer.train_method == "backprop")
            PMGDlayer.layer.weight .-= PMGDlayer.M_int[1] #update weights
            PMGDlayer.layer.bias   .-= PMGDlayer.M_int[2] #update biases
        end
        if (PMGDlayer.train_method == "node")
            PMGDlayer.layer[1].weight .-= PMGDlayer.M_int[1] #update weigths
            PMGDlayer.layer[1].bias   .-= PMGDlayer.M_int[2] #update biases 
        end
    end
    if optimizer=="adam"
        PMGDlayer.M_int[1]  .= beta1*PMGDlayer.M_int[1]   .+ (1-beta1)*PMGDlayer.G_int[1]/tau_theta
        PMGDlayer.M2_int[1] .= beta2*PMGDlayer.M2_int[1] .+ (1-beta2)*(PMGDlayer.G_int[1]/tau_theta).^2

        PMGDlayer.M_int[2]  .= beta1*PMGDlayer.M_int[2]   .+ (1-beta1)*PMGDlayer.G_int[2]/tau_theta
        PMGDlayer.M2_int[2] .= beta2*PMGDlayer.M2_int[2] .+ (1-beta2)*(PMGDlayer.G_int[2]/tau_theta).^2

        m1_hat1=PMGDlayer.M_int[1]
        m1_hat2=PMGDlayer.M_int[2]

        m2_hat1=PMGDlayer.M2_int[1]
        m2_hat2=PMGDlayer.M2_int[2]

        """ Update parameteres in place, needs train_method"""
        if (PMGDlayer.train_method == "weight") || (PMGDlayer.train_method == "backprop")
            PMGDlayer.layer.weight .-= eta*m1_hat1 ./ (sqrt.(m2_hat1) .+ 1e-8) #update weights
            PMGDlayer.layer.bias   .-= eta*m1_hat2 ./ (sqrt.(m2_hat2) .+ 1e-8) #update biases
        end
        if (PMGDlayer.train_method == "node")
            PMGDlayer.layer[1].weight .-= eta*m1_hat1 ./ (sqrt.(m2_hat1) .+ 1e-8) #update weights
            PMGDlayer.layer[1].bias   .-= eta*m1_hat2 ./ (sqrt.(m2_hat2) .+ 1e-8) #update biases
        end
    end

end

function min_weight(w_min,w_update)
    """an utility function that sets minimum update"""
    return max.(w_min,abs.(w_update)) .* sign.(w_update)
end

function norm_from_feval(f_eval)    
    """returns the norm, maximum and minimum values of networks thetas"""
    Flux.params(f_eval)
    return norm(Flux.params(f_eval)), maximum([maximum(n) for n in Flux.params(f_eval)]), minimum([minimum(n) for n in Flux.params(f_eval)])
end

function norm_from_Struct(Struct_Chain)    
    """returns the norm, maximum and minimum values of networks thetas"""
    weigths=[]
    for struct_layer in Struct_Chain
        if (struct_layer.train_method=="weight") || (struct_layer.train_method=="backprop")
            push!(weigths,struct_layer.layer.weight)
            push!(weigths,struct_layer.layer.bias)
        end
        if struct_layer.train_method=="node"
            push!(weigths,struct_layer.layer[1].weight)
            push!(weigths,struct_layer.layer[1].bias)
        end
    end
    return norm(weigths), maximum([maximum(n) for n in weigths]), minimum([minimum(n) for n in weigths])
end

function f_dtheta!(theta_ac::CuArray{Float32}, rng::CUDA.CURAND.RNG)
    # Update theta_ac inplace
    Random.rand!(rng,theta_ac)
    theta_ac .= theta_ac .> 0.5
    theta_ac .-= 0.5
    theta_ac .*= 2
end

function stack_arrays(V)
    """Takes a vector (1D) of arrays (ND) V and combines them into a single block array (N+1)D"""
    new_dimensions = tuple(size(V[1])...,length(V))
    output = zeros(eltype(V[1]), new_dimensions)
    last_dimension = length(new_dimensions)
    for n in 1:length(V)
        selectdim(output, last_dimension, n) .= V[n]
    end
    return output
end

function parameter_combinations(;parameters_named_tuple...)
    """Generate pair-wise combinations of the given parameters"""
    parameter_dict_list = []
    # If any strings present, don't split them up into individual characters
    parameters_dict = Dict(pairs(parameters_named_tuple))
    for (k,v) in parameters_dict
        if v isa String
            parameters_dict[k] = [v]
        end
    end
    # Create cartesian product of all parameter sweeps
    parameter_values = vec(collect(Base.Iterators.product(values(parameters_dict)...)))
    for pv = parameter_values
        pd = Dict(zip(keys(parameters_dict), pv))
        push!(parameter_dict_list, pd)
    end
    # Convert to dataframe and return
    df = DataFrame(parameter_dict_list)
    return df
end


########################################
"""Saving / loading data"""
########################################

function create_results_subfolder(name = "vary-unnamed", i=1)
    results_path = joinpath("results", name)
    new_folder = lpad(i,4,"0")
    path = joinpath(results_path, new_folder)
    if !ispath(path)
        mkpath(path)
    end
    return path
end

function save_network(Struct_Chain,results_path)
    """Save network architecture: each layer type, size and training methods in a txt file"""
    for n in range(1,length(Struct_Chain))
        print_to_text(results_path, "Struct_Chain.txt"; 
        text=" layer "*string(n)*" train method: "*string(Struct_Chain[n].train_method),)
        if Struct_Chain[n].train_method == "node"
            print_to_text(results_path, "Struct_Chain.txt"; 
            text=" layer "*string(n)*" layer: "*string(Struct_Chain[n].layer[1]),)
            print_to_text(results_path, "Struct_Chain.txt"; 
            text=" layer "*string(n)*" layer: "*string(Struct_Chain[n].layer[2].σ),)
        else
            print_to_text(results_path, "Struct_Chain.txt"; 
            text=" layer "*string(n)*" layer: "*string(Struct_Chain[n].layer),)
        end
    end
end

function save_function_parameters(path, filename = "parameters.txt"; kwargs...)
    """ Takes an arbitrary list of function arguments and saves it to a
    TOML-based parameters.txt file at the `path` location.  Called like 
    save_function_parameters("./"; a=5, b=2, c="hello") """
    file_path = joinpath(path, filename)
    open(file_path, "a") do io
        TOML.print(io, kwargs)
    end
end

function save_logger(logger, path, filename = "logger.jld2")
    # Save as JLD2 with compression
    filepath = joinpath(path,filename)
    jldsave(filepath, true; logger = logger)
    return filepath
end

function load_logger(path, filename = "logger.jld2")
    # Save as JLD2 with compression
    filepath = joinpath(path,filename)
    logger = jldopen(filepath)["logger"]
    return logger
end

"""Sequential printing script"""
function print_to_text(path, filename = "debug.txt"; text)
    file_path = joinpath(path, filename)
    open(file_path, "a") do file
        write(file, text)
        write(file, "\n")
    end
end


function run_mgd(;
    Struct_Chain,
    f_eval,
    f_cost,
    f_cost_log,
    f_dtheta!,
    accuracy,
    num_steps,
    tau_p,
    tau_x,
    tau_theta,
    eta,
    optimizer,
    gamma,
    beta1,
    beta2,
    Xtrain,
    Ytrain,
    Xtest,
    Ytest,
    rng,
    batch_size,
    results_path,
    )

    #initialize logger
    logger = Dict()
    #### Intialize variables
    samples_to_use=Random.rand(1:size(Xtrain)[end],batch_size) #choose random sunset of samples
    x=select_samples(Xtrain, samples_to_use)
    y_target=select_samples(Ytrain, samples_to_use)
    y=deepcopy.(y_target)

    # Initialization
    tau_theta_next = tau_theta
    tau_p_next = tau_p
    tau_x_next = tau_x
    start_time = time()

    angle = 0
    C_test = 0
    C = zeros(1,batch_size)
    C0 = zeros(1,batch_size)
    C_ac = zeros(1,batch_size)
    accuracy_test = 0
    accuracy_train=Float32(0) 
    norm_theta = 0
    norm_Gint = 0

    max_accuracy=Float32(0) 
    stable_accuracy_count=0

    min_angle=Float32(90) 
    stable_angle_count=0

    if typeof(eta) <: Real
        eta_array = fill(Float32(eta), 1+num_steps)
    elseif typeof(eta) <: Vector
        eta_array = Float32.(eta)
    else
        error("Eta must either be a number or a 1D array of length num_steps")
    end

    backprop_flag=false
    for PMGDlayer in Struct_Chain
        if PMGDlayer.train_method == "backprop"
            backprop_flag=true
        end
    end

    eta_array[1:1000].=0
    for n in 1:num_steps
        #initialize logger
        #save data and initialize logger
        if (n%10000)==1
            save_index=n÷10000
            purged_logger = Dict()
            for (key, value) in logger
                if length(value) > 0
                    purged_logger[key] = value
                end
            end
            dftosave = DataFrame(purged_logger)
            CSV.write(joinpath(results_path,"data"*string(save_index)*".csv"), dftosave)

            #reinitialize logger
            logger = Dict()
            logger["iteration"] = Int[]
            logger["accuracy_test"] = Float64[]
            logger["accuracy_train"] = Float64[]
            logger["C_test"] = Float64[]
            logger["C_train"] = Float64[]
            logger["eta"] = Float64[]
            logger["time"] = Float64[]
        end

        #### WMGD
        # If more than tau_x has elapsed, update input/target
        if n >= tau_x_next
            samples_to_use=Random.rand(1:size(Xtrain)[end],batch_size)
            x=select_samples(Xtrain, samples_to_use)
            y_target=select_samples(Ytrain, samples_to_use)
            tau_x_next += tau_x
        end

        #evaluate the output
        activations_list=Flux.activations(f_eval,x)
        Struct_Chain[1].x.=x
        for (layer_num, y_temp) in enumerate(activations_list[1:end-1])
            Struct_Chain[layer_num+1].x.=y_temp
        end
        y = f_eval(x) #don't use y=activations_list[end], it won't work

        # If more than tau_p has elapsed, update perturbations (without using memory)
        if (n >= tau_p_next)
            for PMGDlayer in Struct_Chain
                if (PMGDlayer.train_method != "none") && (PMGDlayer.train_method != "backprop")
                    for i in 1:length(PMGDlayer.theta_ac)
                        f_dtheta!(PMGDlayer.theta_ac[i], rng) # Update the theta_ac perturbation direction (+1 ro -1)
                    end
                end
            end
            tau_p_next += tau_p
        end
        
        # for each layer:
        # measure base cost C0
        # apply perturbation
        # measure new cost C
        # calculate change in Cost (C_ac=C-C0)
        # revert the perturbation
        # update G
        
        if backprop_flag
            # Use Flux's gradient() function to get 
            gs = gradient(() -> mean(f_cost(f_eval(x), y_target)), Flux.params(f_eval))
            for PMGDlayer in Struct_Chain
                if (PMGDlayer.train_method != "none")
                    PMGDlayer.G[1].= gs[PMGDlayer.theta_per[1]]
                    PMGDlayer.G[2].= gs[PMGDlayer.theta_per[2]]
                    PMGDlayer.G_int[1] .= PMGDlayer.G[1]
                    PMGDlayer.G_int[2] .= PMGDlayer.G[2]        
                end
            end
        end
        
        for (struct_num, PMGDlayer) in enumerate(Struct_Chain)            
            if (PMGDlayer.train_method != "none") && (backprop_flag == false)           
                # Calculate C_0
                C0 = f_cost(f_eval(x), y_target) #size(1,batchsize)

                # Add perturbation to theta (without using memory)
                for i in 1:length(PMGDlayer.theta_per)
                    PMGDlayer.theta_per[i] .+= PMGDlayer.theta_ac[i] .* PMGDlayer.amplitude_per
                end
                
                C = f_cost(f_eval(x), y_target) #size(1,batchsize), single cost value for each x-y pair in the batch
                C_ac = C-C0  # Filter the cost by subtracting a constant offset C0, size(1,batchsize)

                # Remove perturbation from theta (without using memory)
                for i in 1:length(PMGDlayer.theta_per)
                    PMGDlayer.theta_per[i] .-= PMGDlayer.theta_ac[i] .* PMGDlayer.amplitude_per
                end

                update_G!(PMGDlayer,C_ac,tau_theta)
            end
        end

        #reporting accuracy and cost
        if (n<1000) || (n%100==0)
            push!(logger["iteration"], n) 
            push!(logger["time"], time()-start_time)

            @show n
            @show accuracy_train = accuracy(Xtrain, Ytrain)
            @show accuracy_test = accuracy(Xtest, Ytest)
            @show C_train = mean(f_cost_log(Xtrain, Ytrain))
            @show C_test = mean(f_cost_log(Xtest, Ytest))
    
            push!(logger["eta"], eta_array[n])
            push!(logger["accuracy_test"], accuracy_test)
            push!(logger["accuracy_train"], accuracy_train)
            push!(logger["C_test"], C_test)
            push!(logger["C_train"], C_train)
        end

        # Update weights if more than tau_theta has elapsed, or doing lowpass-style weight updates
        if (n >= tau_theta_next)
            # Update θ parameters/weights
            for PMGDlayer in Struct_Chain
                if (PMGDlayer.train_method != "none") 
                    update_theta!(PMGDlayer,eta_array[n], tau_theta, optimizer, gamma, beta1, beta2)
                    # Reset the integrated gradient
                    for i in 1:length(PMGDlayer.G_int)
                        PMGDlayer.G_int[i] .*= 0
                    end
                end
            end
            tau_theta_next += tau_theta
        end

    end
    return logger
end

#set experiment parameters
experiment_name = "FMNIST"
seed = 0
num_steps=Int64(1e6)

include_class=[0,1,2,3,4,5,6,7,8,9]
label_smoothing = 0.01
batch_size = 100

n_ch=24
cost_type = "crossentropy"

train_method_conv="weight"
train_method_dense="weight"
eta = 0.001
optimizer = "adam" #["vanilla","adam"]
gamma=0
beta1=0.9
beta2=0.999
norm_mag=0.01
tau_p = 1
tau_x = 1000
tau_theta = 1000

#Create a random number generator
rng = CURAND.default_rng()
CUDA.seed!(seed)

#load the dataset
Xtrain, Ytrain, Xtest, Ytest = get_fashion_mnist_dataset(include_class = include_class, label_smoothing = label_smoothing)
samples_to_use=Random.rand(1:size(Xtrain)[end],batch_size)
cur_x=deepcopy.(select_samples(Xtrain, samples_to_use))

#create the network
Struct_Chain=[]

push!(Struct_Chain,PMGDify(
    Conv((3,3), size(cur_x)[3]=>Int64(1*n_ch), stride=1, pad=SamePad(), tanh_fast) |> gpu,
    train_method_conv, norm_mag, rng, cur_x)); cur_x=Struct_Chain[end].layer(cur_x)
push!(Struct_Chain,PMGDify(
    Conv((3,3), size(cur_x)[3]=>Int64(1*n_ch), stride=1, pad=SamePad(), tanh_fast) |> gpu,
    train_method_conv, norm_mag, rng, cur_x)); cur_x=Struct_Chain[end].layer(cur_x)    
push!(Struct_Chain,PMGDify(
    MaxPool((2,2))|> gpu,
    "none",norm_mag, rng,cur_x)); cur_x=Struct_Chain[end].layer(cur_x)
push!(Struct_Chain,PMGDify(
    Conv((3,3), size(cur_x)[3]=>Int64(2*n_ch), stride=1, pad=SamePad(), tanh_fast) |> gpu,
    train_method_conv, norm_mag, rng, cur_x)); cur_x=Struct_Chain[end].layer(cur_x)
push!(Struct_Chain,PMGDify(
    Conv((3,3), size(cur_x)[3]=>Int64(2*n_ch), stride=1, pad=SamePad(), tanh_fast) |> gpu,
    train_method_conv, norm_mag, rng, cur_x)); cur_x=Struct_Chain[end].layer(cur_x)    
push!(Struct_Chain,PMGDify(
    MaxPool((2,2))|> gpu,
    "none",norm_mag, rng,cur_x)); cur_x=Struct_Chain[end].layer(cur_x)
push!(Struct_Chain,PMGDify(
    Conv((3,3), size(cur_x)[3]=>Int64(4*n_ch), stride=1, pad=SamePad(), tanh_fast) |> gpu,
    train_method_conv, norm_mag, rng, cur_x)); cur_x=Struct_Chain[end].layer(cur_x)
push!(Struct_Chain,PMGDify(
    Conv((3,3), size(cur_x)[3]=>Int64(4*n_ch), stride=1, pad=SamePad(), tanh_fast) |> gpu,
    train_method_conv, norm_mag, rng, cur_x)); cur_x=Struct_Chain[end].layer(cur_x)    
push!(Struct_Chain,PMGDify(
    MaxPool((2,2))|> gpu,
    "none",norm_mag, rng,cur_x)); cur_x=Struct_Chain[end].layer(cur_x)
push!(Struct_Chain,PMGDify(
    Flux.flatten |> gpu,
    "none", norm_mag, rng,cur_x)); cur_x=Struct_Chain[end].layer(cur_x)
push!(Struct_Chain,PMGDify(
    Dense(size(cur_x)[1], 4*n_ch, tanh_fast) |> gpu,
    train_method_dense, norm_mag, rng,cur_x)); cur_x=Struct_Chain[end].layer(cur_x)
push!(Struct_Chain,PMGDify(
    Dense(size(cur_x)[1], 4*n_ch, tanh_fast) |> gpu,
    train_method_dense, norm_mag, rng,cur_x)); cur_x=Struct_Chain[end].layer(cur_x)
push!(Struct_Chain,PMGDify(
    Dense(size(cur_x)[1], length(include_class)) |> gpu,
    train_method_dense, norm_mag, rng,cur_x)); cur_x=Struct_Chain[end].layer(cur_x)
    
f_eval=Chain_from_Struct(Struct_Chain) #create the evaluation function

# create function to evaluate the accuracy
accuracy_raw(x, y) = mean(onecold(f_eval(x), 1:length(include_class)) .== onecold(y, 1:length(include_class)))
accuracy(x, y)=mean([accuracy_raw(select_samples(x, samples_to_use),select_samples(y, samples_to_use)) for samples_to_use in partition(1:size(x)[end], 100)]) #if accuracy calculation overloads the GPU, divide data and redo

# create function to evaluate the cost
f_ce(y, y_target) = Flux.Losses.logitcrossentropy(y, y_target,agg=x->sum(x, dims=1))
f_mse(y, y_target) = Flux.Losses.mse(y, y_target,agg=x->sum(x, dims=1)) 
if cost_type == "crossentropy"
    f_cost = f_ce
elseif cost_type == "mse"
    f_cost = f_mse
elseif cost_type == "exp"
    f_cost = f_exp
else
    error("Invalid cost_type")
end

function f_cost_log(x, y)
    """accuracy calculation may overload the GPU, divide data into managaable chunks"""
    return hcat([f_cost(f_eval(select_samples(x, samples_to_use)),select_samples(y, samples_to_use)) for samples_to_use in partition(1:size(x)[end], 100)]...)
end

results_path = create_results_subfolder(experiment_name, 0) # Create a results path to store the data
@show results_path
logger = run_mgd(;
        Struct_Chain = Struct_Chain,
        f_eval=f_eval,
        f_cost = f_cost,
        f_cost_log=f_cost_log,
        f_dtheta! = f_dtheta!,
        accuracy = accuracy,
        num_steps = num_steps,
        tau_p = tau_p,
        tau_x = tau_x,
        tau_theta = tau_theta,
        eta = eta,
        optimizer = optimizer,
        gamma=gamma,
        beta1=beta1,
        beta2=beta2,
        Xtrain = Xtrain,
        Ytrain = Ytrain,
        Xtest = Xtest,
        Ytest = Ytest,
        rng = rng,
        batch_size=batch_size,
        results_path=results_path,
        )

iteration=logger["iteration"]
accuracy_train=logger["accuracy_train"]
accuracy_test=logger["accuracy_test"]
C_train=logger["C_train"]
C_test=logger["C_test"]
