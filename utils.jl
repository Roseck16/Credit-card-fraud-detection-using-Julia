
"""
function check_consistent_length(arrays...)

    Checks that the input arrays have the same lengths.

    Parameters
    -------------------
    arrays...: Array_like


    Examples
    -------------------
    julia> check_consistent_length([1,2,3], [4,5,6])
    julia>

    julia> a = [0.3, 0.1, 0.1, 0.0]
    4-element Array{Float64,1}:
     0.3
     0.1
     0.1
     0.0
    julia> b = [0, 0, 0]
    3-element Array{Int64,1}:
     0
     0
     0
    julia> check_consistent_length(a, b)
    ERROR: Found input variables with inconsistent number of samples: [4, 3]

"""
function check_consistent_length(arrays...)
    lengths = [length(array) for array in arrays]
    uniques = unique(lengths)
    if length(uniques) > 1
        error("Found input variables with inconsistent number of samples: ",
        "$([i for i in lengths])")
    end
end

"""
function assert_all_finite(array::AbstractArray)

    Check if all the elements in an array are finite.

    Parameters:
    -------------------
    array: Array_like


    Examples:
    -------------------
    julia> assert_all_finite([1,2,3])
    julia>

    julia> assert_all_finite([1,2,3, NaN])
    ERROR: Array contains NaN elements
"""
function assert_all_finite(array::AbstractArray)
    for element in array
        if ~isfinite(element)
            error("Array contains NaN elements")
        end
    end
end

"""
function argsort(a::AbstractArray, sort_alg=nothing, reversed::Bool=false)
    It sorts an array and returns the indexes of the
    original array.

    Parameters
    -------------------
    a: array_like
       Array to sort.

    sort_alg: optional
       Algorithm to use to sort the array. The
       default is MergeSort.

    reversed: Bool
       Whether the output array containing the indexes
       is in reversed order or not.


     Examples:
     -------------------

    julia> array = [3, 2, 4 ,1]
    julia> argsort(array)
    4-element Array{Any,1}:
     4
     2
     1
     3

"""
function argsort(a::AbstractArray; sort_alg=nothing, reversed::Bool=false)
    if isnothing(sort_alg)
        _sort_alg = MergeSort
    elseif sort_alg == "MergeSort"
        _sort_alg = MergeSort
    end

    a_copy = sort(a, alg=_sort_alg)
    i = 1
    pairs_list = []
    for element in a
        push!(pairs_list, element => i)
        i += 1
    end

    index_list = []
    for element in a_copy
        for pair in pairs_list
            if pair.first == element
                push!(index_list, pair.second)
            end
        end
    end
    if reversed
        return reverse(index_list)
    else
        return index_list
    end
end

"""
function _where_(array::AbstractArray)

    Returns a list containing the indices for where the
    element of the input array is not 0.

    Parameters:
    -------------------
    array: Array_like


    Examples:
    -------------------
    julia> _where_([1,2,3])
    3-element Array{Any,1}:
     1
     2
     3

    julia> array = [1, 1, -1, -1, 0.5, 0.5, 0, 0.5, 1, -1]
    10-element Array{Float64,1}:
      1.0
      1.0
     -1.0
     -1.0
      0.5
      0.5
      0.0
      0.5
      1.0
     -1.0
     julia> _where_(array)
     9-element Array{Any,1}:
      1
      2
      3
      4
      5
      6
      8
      9
     10

"""
function _where_(array::AbstractArray)
    list = []
    i = 1
    for a in array
        if a != 0
            push!(list, i)
        end
        i += 1
    end
    return list
end

"""
function concatenate_arrays(vargs...)

    Takes various arguments, so far arrays of shape 1 and
    type Int, Float32 or Float64; Numbers; Bools; and concatenates them
    into a single array.

    Examples:
    -------------------

    julia> concatenate_arrays([1,2,3], 4, 5, true, false, [true, false])
    9-element Array{Any,1}:
     1
     2
     3
     4
     5
  true
 false
  true
 false

 julia> concatenate_arrays([1 2; 3 4], [5 6; 7 8])
 ERROR: Element not supported: Array{Int64,2}
"""
function concatenate_arrays(vargs...)
    array = []
    for element in vargs
        if typeof(element) <: Array{Int,1} ||
            typeof(element) <: Array{Float32, 1} ||
            typeof(element) <: Array{Float64, 1} ||
            typeof(element) <: Array{Bool, 1} ||
            typeof(element) <: Array{Any, 1}
            for item in element
                push!(array, item)
            end
        elseif typeof(element) <: Number
            push!(array, element)
        elseif typeof(element) <: Bool
            push!(array, element)
        else
            error("Element not supported: $(typeof(element))")
        end
    end
    return array
end

"""
function logical_or(x::AbstractArray, y::AbstractArray)

    Returns an array or a matrix containing the logical operator or over the arrays x and y.

    Parameters:
    -------------------
    x: Array_like
    y: Array_like


    Examples:
    -------------------
    julia> logical_or([true, true, false], [false, false, false])
    3-element Array{Any,1}:
      true
      true
     false

    julia> logical_or([1 0; 0 1], [1 0; 1 1])
    2×2 Array{Any,2}:
     true  false
     true   true
"""
function logical_or(x::AbstractArray, y::AbstractArray)
    check_consistent_length(x, y)

    array = []
    for i in zip(x, y)
        if typeof(i) <: Tuple{Bool,Bool}
            if i[1] == true || i[2] == true
                push!(array, true)
            else
                push!(array, false)
            end
        elseif typeof(i) <: Tuple{Number, Number}
            if i[1] != 0 || i[2] != 0
                push!(array, true)
            else
                push!(array, false)
            end
        end
    end
    if length(array) == 1
        return array[1]
    else
        if size(x,2) != 1
            return reshape(array, size(x,1), size(x,2))
        else
            return array
        end
    end
end

"""
function custom_diff(A::AbstractArray; n::Int=1, dims::Int=1)

    Variance of the base function diff. The number of times the difference is applied can be selected.

    Parameters:
    -------------------
    A: Array_like
    Array that is going to be differentiated.
    n: Int. Defaults to 1.
    dims: Int. Defaults to 1.


    Examples:
    -------------------
    julia> array = [1,3,7,14,25,34,60]
    7-element Array{Int64,1}:
      1
      3
      7
     14
     25
     34
     60
    julia> custom_diff(array, n=1)
    6-element Array{Int64,1}:
      2
      4
      7
     11
      9
     26
    julia> custom_diff(array, n=2)
    5-element Array{Int64,1}:
      2
      3
      4
     -2
     17
"""
function custom_diff(A::AbstractArray; n::Int=1, dims::Int=1)
    for i = 1:n
        A = diff(A, dims=dims)
    end
    return A
end

"""
function auc(x, y)

    Compute Area Under the Curve (AUC) using the trapezoidal rule. Based in the scikit-learn's function auc.
"""
function custom_auc(x, y)

    x = collect(Iterators.flatten(x))
    y = collect(Iterators.flatten(y))

    direction = 1
    dx = diff(x)
    if any(i -> (i<0), dx)
        if all(i -> (i<=0), dx)
            direction = -1
        else
            error("x is neither increasing nor decreasing ",
                ": $x")
        end
    end
    area = direction .* trapz(y, x)
end

function trapz(y, x=nothing, dx=1.0, axis=nothing)
    if isnothing(x)
        d = dx
    else
        if ndims(x) == 1
            d = diff(x)
            shape = [1 for i = 1:ndims(y)]
            if isnothing(axis)
                shape[end] = size(d, 1)
            else
                shape[axis] = size(d, 1)
            end
            if length(shape) == 1
                d = reshape(d, shape[1], :)
            elseif length(shape) == 2
                d = reshape(d, shape[1], shape[2])
            else
                @warn("Dimensions of y bigger than 3.
                        Only 1 and 2 are supported")
                exit()
            end
        else
            d = diff(x, dims=axis)
        end
    end
    nd = ndims(y)
    slice1 = [range(1, stop=length(y), step=1) for i = 1:nd]
    slice2 = [range(1, stop=length(y), step=1) for i = 1:nd]
    if isnothing(axis)
        slice1[end] = range(1+1, stop=length(y), step=1)
        slice2[end] = range(1, stop=length(y)-1, step=1)
    else
        slice1[axis] = range(1+1, stop=length(y), step=1)
        slice2[axis] = range(1, stop=length(y)-1, step=1)
    end
    ret = sum((d .* (y[slice1[1]] .+ y[slice2[1]]) ./ 2.0), dims=size(y,2))[1]
end

function _binary_clf_curve(y_true, y_score; pos_label=nothing, sample_weight=nothing)
    check_consistent_length(y_true, y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if isnothing(pos_label)
        pos_label = 1
    end

    y_true = (y_true .== pos_label)

    desc_score_indices = argsort(y_score, sort_alg="MergeSort", reversed=true)
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    if ~isnothing(sample_weight)
        weight = sample_weight[desc_score_indices]
    else
        weight = 1.0
    end

    distinct_value_indices = _where_(diff(y_score))
    threshold_indx = concatenate_arrays(distinct_value_indices, size(y_true, 1))

    tps = cumsum(y_true .* weight)[threshold_indx]
    if ~isnothing(sample_weight)
        fps = cumsum((1 .- y_true) .* weight)[threshold_indx]
    else
        fps = threshold_indx .- tps
    end
    return fps, tps, y_score[threshold_indx]
end

"""
function roc_curve(y_true, y_score, pos_label=nothing,
                    sample_weight=nothing, drop_intermediate=true)

    Computes the Receiver operating characteristic (ROC). Based in the scikit-learn's function roc_curve.
"""
function roc_curve(y_true, y_score; pos_label=nothing,
    sample_weight=nothing, drop_intermediate=true)

    fps, tps, thresholds = _binary_clf_curve(y_true, y_score, pos_label=pos_label,
                                            sample_weight=sample_weight)

    if drop_intermediate && length(fps) > 2
        optimal_indxs = _where_(concatenate_arrays(true,
                                        logical_or(custom_diff(fps, n=2),
                                                    custom_diff(tps, n=2)),
                                                    true))
        fps = fps[optimal_indxs]
        tps = tps[optimal_indxs]
        thresholds = thresholds[optimal_indxs]
    end
    tps = concatenate_arrays(0, tps)
    fps = concatenate_arrays(0, fps)
    thresholds = concatenate_arrays(thresholds[1] .+ 1, thresholds)

    if fps[end] <= 0
        #@warn("No negative samples in y_true, false positive value should be meaningless")
        fpr = [NaN for i = 1:size(fps,1)]
    else
        fpr = fps / fps[end]
    end

    if tps[end] <= 0
        #@warn("No positive samples in y_true, true positive value should be meaningless")
        tpr = [NaN for i = 1:size(tps,1)]
    else
        tpr = tps / tps[end]
    end

    return fpr, tpr, thresholds
end
