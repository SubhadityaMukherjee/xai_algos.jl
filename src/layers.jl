# Refs : https://github.com/adrhill/ExplainableAI.jl/
#%%
# LAYER DEFINITIONS
"""
activation(layer)

Return activation function of the layer.
In case the layer is unknown or no activation function is found, `nothing` is returned.
"""
activation(layer) = nothing
activation(l::Dense)         = l.σ
activation(l::Conv)          = l.σ
activation(l::CrossCor)      = l.σ
activation(l::ConvTranspose) = l.σ
activation(l::BatchNorm)     = l.λ
#%%
## Layer types
"""Union type for convolutional layers."""
const ConvLayer = Union{Conv,ConvTranspose,CrossCor}

"""Union type for dropout layers."""
const DropoutLayer = Union{Dropout,typeof(Flux.dropout),AlphaDropout}

"""Union type for reshaping layers such as `flatten`."""
const ReshapingLayer = Union{typeof(Flux.flatten),typeof(Flux.MLUtils.flatten)}

"""Union type for max pooling layers."""
const MaxPoolLayer = Union{MaxPool,AdaptiveMaxPool,GlobalMaxPool}

"""Union type for mean pooling layers."""
const MeanPoolLayer = Union{MeanPool,AdaptiveMeanPool,GlobalMeanPool}

"""Union type for pooling layers."""
const PoolingLayer = Union{MaxPoolLayer,MeanPoolLayer}

# Activation functions
"""Union type for ReLU-like activation functions."""
const ReluLikeActivation = Union{typeof(relu),typeof(gelu),typeof(swish),typeof(mish)}

"""Union type for softmax activation functions."""
const SoftmaxActivation = Union{typeof(softmax),typeof(softmax!)}

# Layers & activation functions supported by LRP
"""Union type for layers that are allowed by default in "deep rectifier networks"."""
const LRPSupportedLayer = Union{Dense,ConvLayer,DropoutLayer,ReshapingLayer,PoolingLayer}

"""Union type for activation functions that are allowed by default in "deep rectifier networks"."""
const LRPSupportedActivation = Union{typeof(identity),ReluLikeActivation}


abstract type AbstractNeuronSelector end

function mask_output_neuron!(
    output_relevance, output_activation, ns::AbstractNeuronSelector
)
    fill!(output_relevance, 0)
    neuron_selection = ns(output_activation)
    output_relevance[neuron_selection] = output_activation[neuron_selection]
end

"""
    MaxActivationSelector()

Neuron selector that picks the output neuron with the highest activation.
"""
struct MaxActivationSelector <: AbstractNeuronSelector end
function (::MaxActivationSelector)(out::AbstractArray{T,N}) where {T,N}
    N < 2 && throw(BATCHDIM_MISSING)
    return Vector{CartesianIndex{N}}([argmax(out; dims=1:(N - 1))...])
end

"""
    IndexSelector(index)

Neuron selector that picks the output neuron at the given index.
"""
struct IndexSelector{I} <: AbstractNeuronSelector
    index::I
end
function (s::IndexSelector{<:Integer})(out::AbstractArray{T,N}) where {T,N}
    N < 2 && throw(BATCHDIM_MISSING)
    return CartesianIndex{N}.(s.index, 1:size(out, N))
end
function (s::IndexSelector{I})(out::AbstractArray{T,N}) where {I,T,N}
    N < 2 && throw(BATCHDIM_MISSING)
    return CartesianIndex{N}.(s.index..., 1:size(out, N))
end

"""
    AugmentationSelector(index)

Neuron selector that passes through an augmented neuron selection.
"""
struct AugmentationSelector{I} <: AbstractNeuronSelector
    indices::I
end
(s::AugmentationSelector)(out) = s.indices