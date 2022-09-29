using Surrogates
using Plots
using Statistics
using Random
using DataFrames
using Distances
using LinearAlgebra
using PolyChaos
default()

# Implementing diversity
function calculate_variance(x, models)
    predictions = []
        for model in models
            prediction = model(x)
            append!(predictions, prediction)
        end
    return var(predictions, corrected=false)
end

function minimum_distance(new_x, prev_x)
    min_dist = Inf
    for point in prev_x
        new_dist = euclidean(point, new_x)
        if new_dist < min_dist
            min_dist = new_dist
        end
    end
    return min_dist
end

function diversity_metric(prev_x, new_x, models, lambda = 0.5, mode=1)
    variance = calculate_variance(new_x, models)
    min_dist = minimum_distance(new_x, prev_x)
    if mode == 0
        return (1 - lambda) * sqrt(variance) + lambda * min_dist
    elseif mode == 1
        return sqrt(variance) * min_dist
    end
end

function query_by_committee(num_iter, num_new_points, sample_space, lower_bound, upper_bound, target_fn, initial_samples)
    prev_points = sample(initial_samples, lower_bound, upper_bound, SobolSample())
    y = target_fn.(x)
    for i in 1:num_iter
        kriging_surrogate = Kriging(prev_points, y, lower_bound, upper_bound, p=1.9)
        my_radial_basis = RadialBasis(prev_points, y, lower_bound, upper_bound)
        x = []
        for i in 1:num_new_points
            max_score = 0
            max_index = -1
            for j in 1:length(sample_space)
                score = diversity_metric(prev_points, sample_space[j], [my_radial_basis, kriging_surrogate], 0, 0)
                if score > max_score
                    max_score = score
                    max_index = j
                end
            end
            append!(x, sample_space[max_index])
            println(sample_space[max_index])
            deleteat!(sample_space, max_index)
        end
        prev_points = vcat(prev_points, x)
        y = f.(prev_points)
    end
    return prev_points, y
end

function visualize()

end
