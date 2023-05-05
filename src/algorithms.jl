function gradient_wrt_input(model, input, output_indices)
    return only(gradient((in) -> model(in)[output_indices], input))
end

function gradients_wrt_batch(model, input::AbstractArray{T,N}, output_indices) where {T,N}
    # To avoid computing a sparse jacobian, we compute individual gradients
    # by calling `gradient_wrt_input` on slices of the input along the batch dimension.
    out = similar(input)
    inds_before_N = ntuple(Returns(:), N - 1)
    for (i, ax) in enumerate(axes(input, N))
        view(out, inds_before_N..., ax, :) .= gradient_wrt_input(
            model, view(input, inds_before_N..., ax, :), drop_batch_index(output_indices[i])
        )
    end
    return out
end

function Gradient(model, input)
    output = model(input)
    output_indices = MaxActivationSelector()(output)
    return gradients_wrt_batch(model, input, output_indices)
end

function InputTimesGradient(model, input)
    output = model(input)
    output_indices = MaxActivationSelector()(output)
    return input .* gradients_wrt_batch(model, input, output_indices)
end