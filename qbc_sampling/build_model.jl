# import Pkg
# Pkg.resolve()

using Surrogates
using Plots
using Statistics
using Random
using DataFrames
using Distances
using LinearAlgebra
using PolyChaos
default()

# Define the 2d Rosenbrock function
function Rosenbrock2d(x)
    x1 = x[1]
    x2 = x[2]
    return (1-x1)^2 + 100*(x2-x1^2)^2
end

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
    min_dist = minimum(broadcast(euclidean, prev_x, [new_x]))
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

# Helper function to get next best point
function next_point_helper(prev_points, sample_space, models, lambda, mode)
    return argmax(broadcast(diversity_metric, Ref(prev_points), sample_space, Ref(models), Ref(lambda), Ref(mode)))
end 



# Main function to add new points to the sample
function query_by_committee(num_iter, num_new_points, sample_space, lb, ub, d, target_fn, initial_samples, lambda, mode)
    prev_points = initial_samples
    y = target_fn.(prev_points)
    for i in 1:num_iter
        kriging_surrogate = Kriging(prev_points, y, lb, ub, p=[1.9 for i in 1:d])
        cub_radial_basis = RadialBasis(prev_points, y, lb, ub, rad=cubicRadial)
        lin_radial_basis = RadialBasis(prev_points, y, lb, ub, rad=linearRadial)
        new_points = []
        for i in 1:num_new_points
            max_index = next_point_helper(prev_points, sample_space, [kriging_surrogate, cub_radial_basis, lin_radial_basis], lambda, mode)
            append!(new_points, sample_space[[max_index]])
            println(sample_space[max_index])
            deleteat!(sample_space, max_index)
        end
        prev_points = vcat(prev_points, new_points)
        y = target_fn.(prev_points)
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
function visualize_2d(prev_points, z, lower_bound, upper_bound)
    new_xs = [xy[1] for xy in prev_points]
    new_ys = [xy[2] for xy in prev_points]
    x = lower_bound[0]:upper_bound[0]
    y = lower_bound[1]:upper_bound[1]
    lin_radial_basis = RadialBasis(prev_points, z, lower_bound, upper_bound, rad=linearRadial)
    cub_radial_basis = RadialBasis(prev_points, z, lower_bound, upper_bound, rad=cubicRadial)
    kriging_surrogate = Kriging(prev_points, z, lower_bound, upper_bound, p=[1.9, 1.9])
    lin_radial_zs = cub_radial_basis.(prev_points)
    cub_radial_zs = cub_radial_basis.(prev_points)
    kriging_zs = kriging_surrogate.(prev_points)
    p1 = surface(x, y, (x, y) -> lin_radial_basis([x y]), title="Radial Basis (Linear)")
    scatter!(new_xs, new_ys, lin_radial_zs)
    
    p2 = surface(x, y, (x, y) -> cub_radial_basis([x y]), title="Radial Basis (Cubic)")
    scatter!(new_xs, new_ys, cub_radial_zs)
    
    p3 = surface(x, y, (x, y) -> kriging_surrogate([x y]), title="Kriging Model")
    scatter!(new_xs, new_ys, kriging_zs)

    p4 = contour(x, y, (x, y) -> lin_radial_basis([x y]), title="Radial Basis (Linear)")
    scatter!(new_xs, new_ys, marker_z=lin_radial_zs)

    p5 = contour(x, y, (x, y) -> cub_radial_basis([x y]), title="Radial Basis (Cubic)")
    scatter!(new_xs, new_ys, marker_z=cub_radial_zs)

    p6 = contour(x, y, (x, y) -> kriging_surrogate([x y]), title="Kriging Model")
    scatter!(new_xs, new_ys, marker_z=kriging_zs)

    display(plot!(p1, p2, p3, p4, p5, p6, size=(1500,700), reuse=false))
end


n = 100     # Number of total sampling points
lb = [0.0,0.0]
ub = [8.0,8.0]
initial_n = 17      # Number of initial sampling points

xys = sample(initial_n,lb,ub,SobolSample());
push!(xys, (0.0, 0.0))
push!(xys, (0.0, 8.0))
push!(xys, (8.0, 0.0))      # Sample from the edges
push!(xys, (8.0, 8.0))
zs = Rosenbrock2d.(xys);

total_n = 100
points_per_iter = 1
total_samples = 2000
orig_train = sample(total_samples, lb, ub, SobolSample())

selected_points, y = query_by_committee(total_n, points_per_iter, orig_train, lb, ub, 2, Rosenbrock2d, xys, 1, 1)