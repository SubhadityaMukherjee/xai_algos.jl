module xai
using Flux
using Metalhead         
using HTTP, FileIO, ImageMagick
using MLUtils
using Colors
using ImageCore
using ImageTransformations: imresize
using ColorSchemes
using Images
using Statistics
using Plots
gr()
using Plots.PlotMeasures

include("utils.jl")
include("layers.jl")
include("algorithms.jl")

# utils
export flatten_model, strip_softmax, copy_layer, has_weight, has_bias, has_weight_and_bias, copy_layer, is_softmax, has_output_softmax, mask_output_neuron!, AbstractNeuronSelector, reduce_method, plot_cam

# layers
export ConvLayer, DropoutLayer, ReshapingLayer, MaxPoolLayer, MeanPoolLayer, PoolingLayer, ReluLikeActivation, SoftmaxActivation, LRPSupportedLayer, LRPSupportedActivation

# algorithms
export gradient_wrt_input, gradients_wrt_batch, Gradient, InputTimesGradient

end