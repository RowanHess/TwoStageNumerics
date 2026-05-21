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
    set_attribute(model, "MemLimit", 16.0)
    set_attribute(model, "TimeLimit", 3500.0)
    set_optimizer_attribute(model, "Threads", 8)
    set_optimizer_attribute(model, "MIPGap", 0.01)
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
    set_attribute(model, "MemLimit", 16.0)
    set_attribute(model, "TimeLimit", 3500.0)
    set_optimizer_attribute(model, "Threads", 8)
    set_optimizer_attribute(model, "MIPGap", 0.01)
    
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
    set_attribute(model, "MemLimit", 16.0)
    set_attribute(model, "TimeLimit", 3500.0)
    set_optimizer_attribute(model, "Threads", 8)

    @variable(model, 0 <= x[1:n, 1:m] <= 1)
    @variable(model, 0 <= y[1:n, 1:m] <= 1)

    @constraint(model, people[i=1:n],
        sum(x[i,j] + y[i,j] for j in 1:m) <= 1)
    
    @constraint(model, houses[j=1:m],
        sum(probs[i,j] * x[i,j] + y[i,j] for i in 1:n) <= 1)
    
    @constraint(model, people2[j=1:m],
        sum(x[i,j] for i in 1:n) <= 1)
    
    @objective(model, Max,
        sum(obj[i,j] * (x[i,j] + y[i,j]) for i in 1:n, j in 1:m))

    optimize!(model)
    return just_obj ? objective_value(model) : (value.(x), value.(y))
end


function one_stage_opt(n, m, probs, obj)

    model = Model(() -> Gurobi.Optimizer(GUROBI_ENV))
    #set_silent(model)
    set_attribute(model, "MemLimit", 16.0)
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
    set_attribute(model, "MemLimit", 16.0)
    set_attribute(model, "TimeLimit", 3500.0)
    set_optimizer_attribute(model, "MIPGap", 0.01)
    set_optimizer_attribute(model, "Threads", 8)
    @variable(model,  x[1:n, 1:m], Bin)
    @variable(model, 0 <= y[1:n, 1:m] <= 1)

    @constraint(model, people[i=1:n],
        sum(x[i,j] + y[i,j] for j in 1:m) <= 1)
    
    @constraint(model, houses[j=1:m],
        sum(probs[i,j] * x[i,j] + y[i,j] for i in 1:n) <= 1)
    
    @constraint(model, people2[j=1:m],
        sum(x[i,j] for i in 1:n) <= 1)
    
    @objective(model, Max,
        sum(obj[i,j] * (x[i,j] + y[i,j]) for i in 1:n, j in 1:m))


    optimize!(model)

    return value.(x), value.(y), objective_value(model)
end


function SAA_no_opt(n, m, probs, obj, s=200)
    scenarios = rand(n, m, s) .< (probs)
    x_val = one_stage_opt(n, m, probs, obj)

    model= Model(() -> Gurobi.Optimizer(GUROBI_ENV))
    #set_silent(model)
    set_attribute(model, "MemLimit", 16.0)
    set_attribute(model, "TimeLimit", 3500.0)
    set_optimizer_attribute(model, "Threads", 8)
    set_optimizer_attribute(model, "MIPGap", 0.01)
    @variable(model, x[1:n, 1:m] , Bin)
    @variable(model, 0<=y[1:n, 1:m, 1:s]<=1)

    @constraint(model, people[i=1:n, scen = 1:s], sum(x[i, :]) + sum(y[i, :, scen]) <= 1)
    @constraint(model, stage1[j=1:m], sum(x[:, j]) <= 1)
    @constraint(model, stage2[j=1:m, scen=1:s],
        sum(x[i,j] for i in 1:n if scenarios[i,j,scen]) +
        sum(y[i,j,scen] for i in 1:n) <= 1)

    obj_s = obj ./ s   # precompute once outside the macro
    @objective(model, Max,
        sum(obj[i,j]   * x[i,j]       for i in 1:n, j in 1:m) +
        sum(obj_s[i,j] * y[i,j,scen]  for i in 1:n, j in 1:m, scen in 1:s))

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

    w0 = vec(sum(x .* obj, dims = 1))

    α = (obj  .- w0').* (y .> 1e-10) .- (y .< 1e-10)
    β = (obj .* p') .* (y .> 1e-10) .- (y .< 1e-10)

    model= Model(() -> Gurobi.Optimizer(GUROBI_ENV))
    @variable(model, 0 <= ϕ[1:n, 1:m])
    @variable(model, 0 <= ψ[1:n, 1:m])
    @constraint(model, house[k=1:m], sum(ϕ[:, k]) + sum(ψ[:, k])<=1)
    @constraint(model, person[i=1:n], sum(ϕ[i, :]) + sum(ψ[i, :])<=1)

    @objective(model, Max, sum(α .* ϕ) + sum(β .* ψ))
    optimize!(model)

    use = sum(value.(ϕ), dims = 1)
    return value.(ϕ) .+ x .* (1 .- use)

end

function find_c_matrix_greedy(d::AbstractVector{<:Real})
    k = length(d)
    c = zeros(Float64, k, k)
    rem = Float64.(copy(d))
    S = sum(rem)
    
    if maximum(d) * 2 > S + 1e-8
        error("Condition max(d) * 2 <= sum(d) violated.")
    end

    while S > 1e-9
        # Get the indices of the remaining needs, sorted highest to lowest
        p = sortperm(rem, rev=true)
        i1, i2 = p[1], p[2]
        v1, v2 = rem[i1], rem[i2]
        
        if v1 < 1e-9; break; end
        
        # Safe amount to allocate without violating the max*2 <= sum invariant for remaining items
        v3 = k >= 3 ? rem[p[3]] : 0.0
        w = min(v1, v2, (S - 2 * v3) / 2)
        
        # Fallback for minor floating point inaccuracies
        if w < 1e-9; w = min(v1, v2); end 
        
        # Distribute the allocation
        c[i1, i2] += w / 2
        c[i2, i1] += w / 2
        
        rem[i1] -= w
        rem[i2] -= w
        S -= 2w
    end
    
    return c
end

function get_abc(weights, p)
    k = length(weights)
    alpha = min.(4/5, 1/5 ./ p)
    beta = max.(0, p .- 2/5)
    gamma = ifelse.(p .> 0, (4/5 .- alpha .- p .* beta) ./ p ./ (1 .- p ./ 2), 0)

    a = alpha .* weights
    b = beta .* weights
    d = gamma .* weights

    if sum(d) >= 2 * maximum(d)
        return (a, b, find_c_matrix_greedy(d))
    else    
        j = argmax(d)
        e = sum(d) - maximum(d)
        g = ifelse.(p .< 2 * p[j], (p[j] .- p) .^ 2, p[j]^2)
        Δ = p[j] * (1-p[j]/2) * e - 0.5 * sum(g .* d)
        b[j] += Δ / p[j]

        d[j] = 0
        c = zeros(k, k)
        c[j, :] = d
        return (a, b, c)        
    end

end

function point_8_from_fluid(n, m, p_input, obj, x, y)
    #x, y = fluid(n, m, probs, obj)

    p = vec(sum(x .* p_input, dims = 1))

    tree = MyTree.build_forest(y)

    function iterate(node, active, parent)
        #println(node)
        if length(node.children) == 0
            return zeros(n, m)
        end
        
        if node.is_row #node is a person
            mod = zeros(n, m)
            if isnothing(parent)
                selection = []
                weights = [child.weight_to_parent for child in node.children]
                probs = [p[child.index] for child in node.children]
                a, b, c = get_abc(weights, probs)
                r = rand()
                for (k, w) in enumerate(a)
                    if r >= 0 && (r -= w) < 0
                        mod[:, node.children[k].index] .-= x[:, node.children[k].index]
                        #println(sum( x[:, node.children[k].index]))
                        mod[node.index, node.children[k].index] += 1
                        push!(selection, k)
                    end
                end
                for (k, w) in enumerate(b)
                    if r >= 0 && (r -= w) < 0
                        push!(selection, k)
                    end
                end
                L = length(b)
                for i=1:L, j=1:L
                    if r >= 0 && (r -= c[i, j]) < 0
                        push!(selection, i)
                        push!(selection, j)
                    end
                end

                for k=1:L
                    mod += iterate(node.children[k], k in selection || rand() > (a[k] + b[k] + sum(c[k, :]) + sum(c[:, k]))/probs[k], nothing)
                end
                
            elseif active
                weights = [node.weight_to_parent, [child.weight_to_parent for child in node.children]...]
                probs = [p[parent], [p[child.index] for child in node.children]...]
                a, b, c = get_abc(weights, probs)
                selection = []
                r = rand() * probs[1]
                if (r -= a[1]) < 0
                        mod[:, parent] .-= x[:, parent]
                        #println(sum(x[:, parent]))
                        mod[node.index, parent] += 1
                        push!(selection, 1)
                elseif (r -= b[1]) < 0
                    push!(selection, 1)
                else
                    for i=1:length(a)
                        if r>= 0 && (r -= c[1, i] + c[i, 1]) < 0
                            push!(selection, i)
                            push!(selection, 1)
                        end
                    end
                    if length(selection) > 0
                        for i=2:length(probs)
                            mod += iterate(node.children[i-1], i in selection || rand() > (a[i] + b[i] + sum(c[i, :]) + sum(c[:, i]))/probs[i], nothing)
                        end
                    else
                        mod += iterate(node, false, parent)
                    end
                end
            else
                weights = [node.weight_to_parent, [child.weight_to_parent for child in node.children]...]
                probs = [p[parent], [p[child.index] for child in node.children]...]
                a, b, c = get_abc(weights, probs)
                selection = []

                o = a[1] + b[1] + sum(c[1, :]) + sum(c[:, 1])
                r = rand() * (1-o)
                L = length(b)

                for k=2:L
                    if r >= 0 && (r -= a[k]) < 0
                        mod[:, node.children[k-1].index] .-= x[:, node.children[k-1].index]
                        #println(sum(x[:, node.children[k-1].index]))
                        mod[node.index, node.children[k-1].index] += 1
                        push!(selection, k)
                    end
                end
                for k=2:L
                    if r >= 0 && (r -= b[k]) < 0
                        push!(selection, k)
                    end
                end

                for i=2:L, j=2:L
                    if r >= 0 && (r -= c[i, j]) < 0
                        push!(selection, i)
                        push!(selection, j)
                    end
                end

                for k=2:L
                    mod += iterate(node.children[k-1], k in selection || rand() > (a[k] + b[k] + sum(c[k, :]) + sum(c[:, k]))/probs[k], nothing)
                end


            end
            return mod

        else #node is a house
            if active
                return sum([iterate(child, false, nothing) for child in node.children])
            end
            r = rand() * (p[node.index] - node.weight_to_parent)
            mod = zeros(n, m)
            for child in node.children
                if r > 0
                    r -= child.weight_to_parent
                    if r <= 0
                        mod += iterate(child, true, node.index)
                    else
                        mod += iterate(child, false, node.index)
                    end
                else
                    mod += iterate(child, false, node.index)
                end
            end
            return mod
        end

    end
    ret = zeros(n, m)
    ret .+= x
    for t in tree
        ret .+= iterate(t, false, nothing)
    end
    return ret
    

    
end

function get_value(sol, scenarios, n, m, probs, obj)
    s = size(scenarios, 3)

    model = Model(() -> Gurobi.Optimizer(GUROBI_ENV))
    set_silent(model)

    @variable(model, 0 <= y[1:n, 1:m] <= 1)

    # RHS is constant across scenarios — computed once
    @constraint(model, people[i=1:n],
        sum(y[i,j] for j in 1:m) <= 1 - sum(sol[i,:]))

    # RHS will be updated per scenario
    @constraint(model, houses[j=1:m],
        sum(y[i,j] for i in 1:n) <= 1.0)

    @objective(model, Max,
        sum(obj[i,j] * y[i,j] for i in 1:n, j in 1:m))

    v = 0.0
    for scen in eachslice(scenarios, dims=3)
        # m-vector: how much capacity each house loses in this scenario
        contrib = vec(sum(sol .* scen, dims=1))
        for j in 1:m
            set_normalized_rhs(houses[j], 1.0 - contrib[j])
        end
        optimize!(model)
        v += objective_value(model)
    end

    return v / s + sum(sol .* obj)
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
     Dict("name" => "one_stage_opt", "requires_fluid" => false, "function" => one_stage_opt),
    Dict("name" => "alternative_sol_straight", "requires_fluid" => false, "function" => alternative_sol_straight),
    Dict("name" => "alt_sol_from_fluid", "requires_fluid" => true, "function" => alt_sol_from_fluid),
    Dict("name" => "point8_sol_straight", "requires_fluid" => false, "function" => point8_sol_straight),
    Dict("name" => "point_8_from_fluid", "requires_fluid" => true, "function" => point_8_from_fluid),
    Dict("name" => "efficient_567", "requires_fluid" => false, "function" => efficient_567),
    Dict("name" => "SAA_no_opt", "requires_fluid" => false, "function" => SAA_no_opt),
    
   
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

        scenarios = rand(n, m, 200) .< (probs)
        for f in functions
            if isfile("$dir/$(f["name"])_sol_$m.txt")
                x = read_sol(m, "$dir/$(f["name"])_sol_$m.txt", n)
                val = get_value(x, scenarios, n, m, probs, obj)
                open("$dir/$(f["name"])_$m.txt", "a") do io
                    println(io, "$(val)")
                end
            end
        end
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

m = parse(Int, ARGS[1])
i = parse(Int, ARGS[2])
if i == 0
    main(m, i)
elseif i < 4
    main(m, 2 * i-1)
    main(m, 2 * i)

elseif i == 4
    main(m, 7)
else
    main(m, 8)
end


#main(300, 5)
# m = 10
# for i = 0:9
#     main(m, i)
# end