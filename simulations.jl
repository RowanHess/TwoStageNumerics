using JuMP, Gurobi, CSV, DataFrames

include("GKPSCompleteBipartite.jl")
using .GKPSCompleteBipartite

include("MyTree.jl")
using .MyTree
ENV["GRB_LICENSE_FILE"] = "../../gurobi.lic"
const GUROBI_ENV = Gurobi.Env()
function alternative_sol_straight(n, m, probs, obj)

    model = Model(() -> Gurobi.Optimizer(GUROBI_ENV))
    #set_silent(model)
    @variable(model, x[1:n, 1:n+1, 1:m], Bin)
    @constraint(model, c[i=1:n, k=1:m], x[i, i, k] == 0)
    @constraint(model, row[i=1:n], sum(x[i, :, :]) + sum(x[:, i, :]) <=1)
    @constraint(model, house[k=1:m], sum(x[:, :, k]) <=1)
    
    real_obj = reshape(obj, (n, 1, m)) .+ (reshape(cat(obj, zeros(m)', dims = 1), (1, n+1, m)) .* (1 .- reshape(probs, (n, 1, m))))
    @objective(model, Max, sum(x .* real_obj))
    optimize!(model)
    #println(objective_value(model))
    return reshape(sum(value.(x), dims = 2), (n, m))

end

function point8_sol_straight(n, m, probs, obj)

    model = Model(() -> Gurobi.Optimizer(GUROBI_ENV))
    #set_silent(model)
    
    # Variables: z_{g, i, j, k, l} mapped to dimensions 1 through 5
    # Dim 1: g ∈ A ∪ {0} -> 1:n+1
    # Dim 2: i ∈ A       -> 1:n
    # Dim 3: j ∈ A ∪ {0} -> 1:n+1
    # Dim 4: k ∈ H       -> 1:m
    # Dim 5: l ∈ H ∪ {0} -> 1:m+1
    @variable(model, x[1:n+1, 1:n, 1:n+1, 1:m, 1:m+1], Bin)
    
    # Constraint 1: Person capacity (Eq. 2)
    @constraint(model, person_cap[a=1:n], 
        sum(x[a, :, :, :, :]) + sum(x[:, a, :, :, :]) + sum(x[:, :, a, :, :]) <= 1
    )
    
    # Constraint 2: House capacity (Eq. 3)
    @constraint(model, house_cap[h=1:m], 
        sum(x[:, :, :, h, :]) + sum(x[:, :, :, :, h]) <= 1
    )
    
    # Pre-pad objective and probabilities with zeros for dummy (0) assignments
    # Dummy persons/houses have 0 value and 0 probability of failing and passing to a backup
    obj_pad = zeros(n+1, m+1)
    obj_pad[1:n, 1:m] .= obj
    
    p_pad = zeros(n+1, m+1)
    p_pad[1:n, 1:m] .= 1 .- probs
    
    # Expand components into 5D shapes for Julia's native broadcasting
    obj_ik = reshape(obj, (1, n, 1, m, 1))
    obj_jl = reshape(obj_pad, (1, 1, n+1, 1, m+1))
    p_ik   = reshape(1 .- probs, (1, n, 1, m, 1))
    
    # For obj_gk, k is dim 4 which only goes up to m
    obj_gk = reshape(obj_pad[:, 1:m], (n+1, 1, 1, m, 1))
    
    p_jl   = reshape(p_pad, (1, 1, n+1, 1, m+1))
    obj_gl = reshape(obj_pad, (n+1, 1, 1, 1, m+1))
    
    # Build the objective coefficient matrix dynamically via broadcasting
    # w_{ik} + w_{j\ell} + p_{ik}w_{gk} + p_{j\ell}(1-p_{ik})w_{g\ell}
    real_obj = obj_ik .+ obj_jl .+ p_ik .* obj_gk .+ p_jl .* (1 .- p_ik) .* obj_gl
    
    @objective(model, Max, sum(x .* real_obj))
    optimize!(model)
    
    println(objective_value(model))
    
    # Return the marginal assignment mapping between `i` and `k` 
    # Sums out dimensions 1 (g), 3 (j), and 5 (l), resulting in a 1 x n x 1 x m x 1 tensor.
    # Reshaping it collapses to the exact n x m probability shape matrix from the original script
    return reshape(sum(value.(x), dims = (1, 3, 5)), (n, m))

end


function linear(n, m, probs, obj, just_obj = true)
    model = Model(() -> Gurobi.Optimizer(GUROBI_ENV))
    #set_silent(model)
    set_attribute(model, "MemLimit", 15.0)
    set_attribute(model, "TimeLimit", 3500.0)
    set_optimizer_attribute(model, "Threads", 8)

    @variable(model, 0<=x[1:n, 1:m]<=1)
    @variable(model, 0<=y[1:n, 1:m]<=1)
    @constraint(model, people[i=1:n], sum(x[i, :]) + sum(y[i, :]) <= 1)
    @constraint(model, houses[j=1:m], sum((x .* probs)[:, j]) + sum(y[:, j]) <= 1)
    @constraint(model, people2[j=1:m], sum(x[:, j])<= 1)
    @objective(model, Max,  sum(y .* obj) + sum(x .* obj))

    optimize!(model)
    if just_obj
        return objective_value(model)
    end

    return value.(x), value.(y)
end


function one_stage_opt(n, m, probs, obj)

    model = Model(() -> Gurobi.Optimizer(GUROBI_ENV))
    #set_silent(model)
    set_attribute(model, "MemLimit", 15.0)
    set_attribute(model, "TimeLimit", 3500.0)
    set_optimizer_attribute(model, "Threads", 8)
    set_optimizer_attribute(model, "MIPGap", 0.01)
    @variable(model, x[1:n, 1:m], Bin)
    @constraint(model, row[i=1:n], sum(x[i, :]) <=1)
    @constraint(model, house[k=1:m], sum(x[:, k]) <=1)
    @objective(model, Max, sum(x .* obj))
    optimize!(model)
    
    return value.(x)

end

function fluid(n, m, probs, obj)
    model = Model(() -> Gurobi.Optimizer(GUROBI_ENV))
    ##set_silent(model)
    set_attribute(model, "MemLimit", 15.0)
    set_attribute(model, "TimeLimit", 3500.0)
    set_optimizer_attribute(model, "MIPGap", 0.01)
    set_optimizer_attribute(model, "Threads", 8)
    @variable(model, x[1:n, 1:m], Bin)
    @variable(model, 0<=y[1:n, 1:m]<=1)
    @constraint(model, people[i=1:n], sum(x[i, :]) + sum(y[i, :]) <= 1)
    @constraint(model, houses[j=1:m], sum((x .* probs)[:, j]) + sum(y[:, j]) <= 1)
    @constraint(model, people2[j=1:m], sum(x[:, j])<= 1)
    @objective(model, Max,  sum(y .* obj) + sum(x .* obj))

    optimize!(model)

    return value.(x), value.(y), objective_value(model)
end


function SAA_no_opt(n, m, probs, obj, s=1000)
    scenarios = rand(n, m, s) .< (probs)
    x_val = one_stage_opt(n, m, probs, obj)

    model= Model(() -> Gurobi.Optimizer(GUROBI_ENV))
    #set_silent(model)
    set_attribute(model, "MemLimit", 15.0)
    set_attribute(model, "TimeLimit", 3500.0)
    set_optimizer_attribute(model, "Threads", 8)
    set_optimizer_attribute(model, "MIPGap", 0.01)
    @variable(model, x[1:n, 1:m] , Bin)
    @variable(model, 0<=y[1:n, 1:m, 1:s]<=1)

    @constraint(model, people[i=1:n, scen = 1:s], sum(x[i, :]) + sum(y[i, :, scen]) <= 1)
    @constraint(model, stage1[j=1:m], sum(x[:, j]) <= 1)
    @constraint(model, stage2[j=1:m, scen = 1:s], sum((x .* scenarios[:, :, scen])[:, j]) + sum(y[:, j, scen]) <= 1)

    @objective(model, Max, sum(x .* obj) + sum(y .* obj) / s)
    set_start_value.(x, x_val)
    optimize!(model)
    #print(objective_value(model))
    return value.(model[:x])

end


function efficient_567(n, m, probs, obj)
    x, y = linear(n, m, probs, obj, false)
    X, _ = GKPSCompleteBipartite.gkps_round_complete(vec(x), n, m)
    Y,_ = GKPSCompleteBipartite.gkps_round_complete(vec(y), n, m)

    X = reshape(X, n, m)
    Y = reshape(Y, n, m)

    x_sum = sum(x, dims = 2)
    y_sum = sum(y, dims = 2)

    x_prime = vec(zeros(n) .+ sum(X, dims = 2))
    y_prime = vec(zeros(n) .+ sum(Y, dims = 2))

    for i=1:n
        if x_prime[i] > 0 && y_prime[i] > 0
            r = rand()
            if r < x_sum[i]
                x_prime[i] = 0
            elseif r < x_sum[i] + y_sum[i]
                y_prime[i] = 0
            else
                x_prime[i] = 0
                y_prime[i] = 0
            end
        end
    end

    p = vec(sum(x .* probs, dims = 1))
    q = rand(m) .< p
    #println(X, Y)
    return X .* (x_prime * q') + Y .* (y_prime * (1 .- q)')
end

function alt_sol_from_fluid(n, m, probs, obj,x, y)
    #x, y = fluid(n, m, probs, obj)

    p = vec(sum(x .* probs, dims = 1))

    mults = min.(3/4, 1/4 ./ p)

    tree = build_forest(y)

    function iterate(node, active)
        #println(node)
        if length(node.children) == 0
            return zeros(n, m)
        end
        if active
            return sum([iterate(child, false) for child in node.children])
        end
        if node.is_row #node is a person
            r = rand() * (1 - node.weight_to_parent)
            mod = zeros(n, m)
            for child in node.children
                if r > 0
                    r -= child.weight_to_parent
                    if r <= 0
                        mod += iterate(child, true)
                        if rand() < mults[child.index] * child.weight_to_parent
                            mod .-= x[:, child.index]
                            mod[node.index, child.index] += 1
                        end
                    else
                        mod += iterate(child, false)
                    end
                else
                    mod += iterate(child, false)
                end
            end
            return mod
        else #node is a house
            r = rand() * (1 - node.weight_to_parent)
            mod = zeros(n, m)
            for child in node.children
                if r > 0
                    r -= child.weight_to_parent
                    if r <= 0
                        mod += iterate(child, true)
                        if rand() < mults[node.index] * child.weight_to_parent
                            mod .-= x[:, node.index]
                            mod[child.index, node.index] += 1
                        end
                    else
                        mod += iterate(child, false)
                    end
                else
                    mod += iterate(child, false)
                end
            end
            return mod

        end

    end
    ret = zeros(n, m)
    ret .+= x
    for t in tree
        ret .+= iterate(t, false)
    end
    return ret
end
function point_8_from_fluid(n, m, probs, obj, x, y)
    #x, y = fluid(n, m, probs, obj)

    p = vec(sum(x .* probs, dims = 1))

    mults = min.(4/5, 1/5 ./ p)

    tree = build_forest(y)

    function iterate(node, active)
        #println(node)
        if length(node.children) == 0
            return zeros(n, m)
        end
        if active
            return sum([iterate(child, false) for child in node.children])
        end
        if node.is_row #node is a person
            r = rand() * (1 - node.weight_to_parent)
            mod = zeros(n, m)
            for child in node.children
                if r > 0
                    r -= child.weight_to_parent
                    if r <= 0
                        mod += iterate(child, true)
                        if rand() < mults[child.index] * child.weight_to_parent
                            mod .-= x[:, child.index]
                            mod[node.index, child.index] += 1
                        end
                    else
                        mod += iterate(child, false)
                    end
                else
                    mod += iterate(child, false)
                end
            end
            return mod
        else #node is a house
            r = rand() * (1 - node.weight_to_parent)
            mod = zeros(n, m)
            for child in node.children
                if r > 0
                    r -= child.weight_to_parent
                    if r <= 0
                        mod += iterate(child, true)
                        if rand() < mults[node.index] * child.weight_to_parent
                            mod .-= x[:, node.index]
                            mod[child.index, node.index] += 1
                        end
                    else
                        mod += iterate(child, false)
                    end
                else
                    mod += iterate(child, false)
                end
            end
            return mod

        end

    end
    ret = zeros(n, m)
    ret .+= x
    for t in tree
        ret .+= iterate(t, false)
    end
    return ret
end

function get_value(sol, scenarios, n, m, probs, obj)
    v = 0
    model = Model(() -> Gurobi.Optimizer(GUROBI_ENV))
    #set_silent(model)
    @variable(model, z[1:n, 1:m])
    @variable(model, 0<=y[1:n, 1:m]<=1)
    @constraint(model, people[i=1:n], sum(sol[i, :]) + sum(y[i, :]) <= 1)
    @constraint(model, houses[j=1:m], sum(z[:, j]) + sum(y[:, j]) <= 1)
    @objective(model, Max,  sum(y .* obj))
    for scen in eachslice(scenarios, dims = 3)

        fix.(model[:z], sol .* (scen))
        optimize!(model)

        v += objective_value(model)

    end

    return v / size(scenarios)[3] + sum(sol .* obj)

end

function generate(m)
    n = Int(floor(m * 1.5))

    obj = zeros(n, m)
    probs = ones(n, m)

    #high value
    for i=1:n
        ms = 1 .+ Int.(floor.(rand(2) .* 1.5 .* m))
        for k in ms
            if k<= m
                obj[i, k]= 1
            end
        end
    end

    #high efficiency
        for i=1:n
        ms = 1 .+ Int.(floor.(rand(2) .* 1.5 .* m))
        for k in ms
            if k<= m
                prob = rand() / 2
                obj[i, k] = min(0.99, (3+rand()) * (prob))
                probs[i, k] = prob
            end
        end
    end
    return n, m, obj, probs

end
function initialize(m)
    n, m, obj, probs = generate(m)

    dir = "results_$m"
    mkpath(dir)

    fluid_data = @timed fluid(n, m, probs, obj)
    touch("$dir/results_$m.txt")
    open("$dir/results_$m.txt", "w") do io
        write(io, "$(fluid_data.time)\n")
        write(io, "fluid_ub: $(fluid_data.value[3])\n")
    end

    df = DataFrame(obj, :auto)
    CSV.write("$dir/obj_$m.csv", df)
    df = DataFrame(probs, :auto)
    CSV.write("$dir/probs_$m.csv", df)

    df = DataFrame(fluid_data.value[1], :auto)
    CSV.write("$dir/fluid_x_$m.csv", df)
    df = DataFrame(fluid_data.value[2], :auto)
    CSV.write("$dir/fluid_y_$m.csv", df)
end

functions = [
    Dict("name" => "alternative_sol_straight", "requires_fluid" => false, "function" => alternative_sol_straight),
    Dict("name" => "point8_sol_straight", "requires_fluid" => false, "function" => point8_sol_straight),
    Dict("name" => "one_stage_opt", "requires_fluid" => false, "function" => one_stage_opt),
    Dict("name" => "SAA_no_opt", "requires_fluid" => false, "function" => SAA_no_opt),
    Dict("name" => "efficient_567", "requires_fluid" => false, "function" => efficient_567),
    Dict("name" => "alt_sol_from_fluid", "requires_fluid" => true, "function" => alt_sol_from_fluid),
    Dict("name" => "point_8_from_fluid", "requires_fluid" => true, "function" => point_8_from_fluid),
]

function write_sol(x, name, m)
    open(name, "w") do io
        for k=1:m
            res = findfirst(c -> c==1, x[:, k])
            if isnothing(res)
                println(io, -1)
            else
                println(io, res)
            end
        end
    end
end

function read_sol(m, name, n)
    x = zeros(n, m)
    k = 1
    for line in eachline(name)
        val = parse(Int, line)
        if val > 0
            x[val, k] = 1
        end
        k += 1
    end
    return x
end

function main(m, index)
    dir = "results_$m"

    if index == 0
        initialize(m)
    elseif index <= length(functions)
        n = Int(floor(m * 1.5))

        obj = Matrix(CSV.read("$dir/obj_$m.csv", DataFrame))
        probs = Matrix(CSV.read("$dir/probs_$m.csv", DataFrame))
        if functions[index]["requires_fluid"]
            x = Matrix(CSV.read("$dir/fluid_x_$m.csv", DataFrame))
            y = Matrix(CSV.read("$dir/fluid_y_$m.csv", DataFrame))
            ft = 0
            open("$dir/results_$m.txt", "r") do io
                ft += parse(Float64, readline(io))
            end
            function_data = @timed functions[index]["function"](n, m, probs, obj, x, y)

            touch("$dir/$(functions[index]["name"])_$m.txt")
            open("$dir/$(functions[index]["name"])_$m.txt", "w") do io
                write(io, "$(function_data.time + ft)\n")
            end
            write_sol(function_data.value, "$dir/$(functions[index]["name"])_sol_$m.txt", m)
        else
            function_data = @timed functions[index]["function"](n, m, probs, obj)

            touch("$dir/$(functions[index]["name"])_$m.txt")
            open("$dir/$(functions[index]["name"])_$m.txt", "w") do io
                write(io, "$(function_data.time)\n")
            end
            write_sol(function_data.value, "$dir/$(functions[index]["name"])_sol_$m.txt", m)
        end
    elseif index == length(functions) + 1
        n = Int(floor(m * 1.5))

        obj = Matrix(CSV.read("$dir/obj_$m.csv", DataFrame))
        probs = Matrix(CSV.read("$dir/probs_$m.csv", DataFrame))

        scenarios = rand(n, m, 1000) .< (probs)
        for f in functions
            if isfile("$dir/$(f["name"])_sol_$m.txt")
                x = read_sol(m, "$dir/$(f["name"])_sol_$m.txt", n)
                val = get_value(x, scenarios, n, m, probs, obj)
                open("$dir/$(f["name"])_$m.txt", "a") do io
                    println(io, "$(val)")
                end
            end
        end
    else
        for f in functions
            if isfile("$dir/$(f["name"])_sol_$m.txt")
                open("$dir/$(f["name"])_$m.txt", "r") do io
                    t = parse(Float64, readline(io))
                    v = parse(Float64, readline(io))
                    open("$dir/results_$m.txt", "a") do io
                        println(io, "$(f["name"]) val: $v")
                        println(io, "$(f["name"]) time: $t")
                    end
                end
            end
        end
    end
end


m = 300#parse(Int, ARGS[1])
i = 0#parse(Int, ARGS[2])

main(m, i)
