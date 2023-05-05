function has_activation(layer)
    hasproperty(layer, :σ) && return true
    hasproperty(layer, :λ) && return true
    return !isnothing(activation(layer))
end

"""
    flatten_model(c)

Flatten a Flux chain containing Flux chains.
"""
function flatten_model(chain::Chain)
    if any(isa.(chain.layers, Chain))
        flatchain = Chain(vcat(_flatten_model.(chain.layers)...)...)
        return flatten_model(flatchain)
    end
    return chain
end

#%%
_flatten_model(x) = x
_flatten_model(c::Chain) = [c.layers...]
is_softmax(x) = x isa SoftmaxActivation
has_output_softmax(x) = is_softmax(x) || is_softmax(activation(x))
has_output_softmax(model::Chain) = has_output_softmax(model[end])

"""
    strip_softmax(model)

Remove softmax activation on model output if it exists.
"""
function strip_softmax(model::Chain)
    if has_output_softmax(model)
        model = flatten_model(model)
        if is_softmax(model[end])
            return Chain(model.layers[1:(end - 1)]...)
        end
        return Chain(model.layers[1:(end - 1)]..., strip_softmax(model[end]))
    end
    return model
end
strip_softmax(l) = copy_layer(l, l.weight, l.bias; σ=identity)

has_weight(layer) = hasproperty(layer, :weight)
has_bias(layer) = hasproperty(layer, :bias)
has_weight_and_bias(layer) = has_weight(layer) && has_bias(layer)

"""
    copy_layer(layer, W, b, [σ=identity])

Copy layer using weights `W` and `b`. The activation function `σ` can also be set,
defaulting to `identity`.
"""
copy_layer(::Dense, W, b; σ=identity) = Dense(W, b, σ)
copy_layer(l::Conv, W, b; σ=identity) = Conv(σ, W, b, l.stride, l.pad, l.dilation, l.groups)
function copy_layer(l::ConvTranspose, W, b; σ=identity)
    return ConvTranspose(σ, W, b, l.stride, l.pad, l.dilation, l.groups)
end
function copy_layer(l::CrossCor, W, b; σ=identity)
    return CrossCor(σ, W, b, l.stride, l.pad, l.dilation)
end
#%%
# Image preprocessing for ImageNet models.
# Code adapted from Metalhead 0.5.3's deprecated utils.jl

# Coefficients taken from PyTorch's ImageNet normalization code
const PYTORCH_MEAN = [0.485f0, 0.456f0, 0.406f0]
const PYTORCH_STD  = [0.229f0, 0.224f0, 0.225f0]
const IMGSIZE      = (224, 224)

# Take rectangle of pixels of shape `outsize` at the center of image `im`
adjust(i::Integer) = ifelse(iszero(i % 2), 1, 0)
function center_crop_view(im::AbstractMatrix, outsize=IMGSIZE)
    im = imresize(im; ratio=maximum(outsize .// size(im)))
    h2, w2 = div.(outsize, 2) # half height, half width of view
    h_adjust, w_adjust = adjust.(outsize)
    return @view im[
        ((div(end, 2) - h2):(div(end, 2) + h2 - h_adjust)) .+ 1,
        ((div(end, 2) - w2):(div(end, 2) + w2 - w_adjust)) .+ 1,
    ]
end

"""
    preprocess_imagenet(img)

Preprocess an image for use with Metalhead.jl's ImageNet models using PyTorch weights.
Uses PyTorch's normalization constants.
"""
function preprocess_imagenet(im::AbstractMatrix{<:AbstractRGB}, T=Float32::Type{<:Real})
    im = center_crop_view(im)
    im = (channelview(im) .- PYTORCH_MEAN) ./ PYTORCH_STD
    return convert.(T, PermutedDimsArray(im, (3, 2, 1))) # Convert Image.jl's CHW to WHC
end

"""
    batch_dim_view(A)

Return a view onto the array `A` that contains an extra singleton batch dimension at the end.
This avoids allocating a new array.

## Example
```juliarepl
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> batch_dim_view(A)
2×2×1 view(::Array{Int64, 3}, 1:2, 1:2, :) with eltype Int64:
[:, :, 1] =
 1  2
 3  4
```
"""
batch_dim_view(A::AbstractArray{T,N}) where {T,N} = view(A, ntuple(Returns(:), N + 1)...)

"""
    drop_batch_index(I)

Drop batch dimension index (last value) from CartesianIndex.

## Example
julia> drop_batch_index(CartesianIndex(5,3,2))
CartesianIndex(5, 3)
"""
drop_batch_index(C::CartesianIndex) = CartesianIndex(C.I[1:(end - 1)])

function reduce_method(attr, method::Symbol)
    if method == :sum
        return Base.reduce(+, attr; dims=3)
    elseif method == :maxabs
        return max.(abs.(attr))
    elseif method == :norm
        return sqrt.(sum(attr .^ 2; dims=3))
    end
end

function plot_cam(attr,method = :norm, size = (224, 224), dpi=300, savepath = nothing)
    tempplot = ColorSchemes.get(colorschemes[:grays], reduce_method(attr, method), :extrema)
    tempplot = plot(tempplot, ticks=false, legend=false, axis=false, size=size, margin = -2mm, dpi =dpi, framestyle = :none, aspect_ratio=:equal)
    if savepath !== nothing
        savefig(tempplot, savepath)
    end
    return tempplot
end