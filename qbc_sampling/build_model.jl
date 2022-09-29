using Surrogates
using Plots
using Statistics
using Random
using DataFrames
using Distances
using LinearAlgebra
using PolyChaos
default()

# Computing the diversity of a new target point by measuring the variance amongst
# the predictions of the models in the committee
function calculate_variance(x, models)
    predictions = []
        for model in models
            prediction = model(x)
            append!(predictions, prediction)
        end
    return var(predictions, corrected=false)
end

# Calculating the minimum distance between a new target point and the previously
# sampled points
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

# Calculating the diversity metric of a new target point
function diversity_metric(prev_x, new_x, models, lambda = 0.5, mode=1)
    variance = calculate_variance(new_x, models)
    min_dist = minimum_distance(new_x, prev_x)
    if mode == 0
        return (1 - lambda) * sqrt(variance) + lambda * min_dist
    elseif mode == 1
        return sqrt(variance) * min_dist
    end
end

# Main function to add new points to the sample
function query_by_committee(num_iter, num_new_points, sample_space, lower_bound, upper_bound, target_fn, initial_samples)
    prev_points = sample(initial_samples, lower_bound, upper_bound, SobolSample())
    y = target_fn.(x)
    for i in 1:num_iter
        kriging_surrogate = Kriging(prev_points, y, lower_bound, upper_bound, p=1.9)
        radial_basis = RadialBasis(prev_points, y, lower_bound, upper_bound)
        x = []
        for i in 1:num_new_points
            max_score = 0
            max_index = -1
            for j in 1:length(sample_space)
                score = diversity_metric(prev_points, sample_space[j], [radial_basis, kriging_surrogate], 0, 0)
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

# Compute the error between the committee's predictions and the actual value
function calculate_error(point, models, actual, mode="MSE")
    target = actual(point)
    errors = []
    for model in models
        prediction = model(point)
        append!(errors, abs(target - prediction))
    end
    if mode == "MSE"
        return mean(errors.^2)
    end
    if mode == "max"
        return maximum(errors)
    end
end

# Visualizing the models' predictions
function visualize_2d(prev_points, y, lower_bound, upper_bound)
    radial_basis = RadialBasis(prev_points, y, lower_bound, upper_bound)
    plot(prev_points, y, seriestype=:scatter, label="Sampled points", xlims=(lower_bound, upper_bound), ylims=(-7, 17), legend=:top)
    plot!(xs, f.(xs), label="True function", legend=:top)
    plot!(xs, radial_basis.(xs), label="Surrogate function", legend=:top)
end
