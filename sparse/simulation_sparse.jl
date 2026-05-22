using SparseArrays
using JuMP
using CSV
using DataFrames
using Gurobi

include("GKPS_sparse.jl")
using .GKPSCompleteBipartite
ENV["GRB_LICENSE_FILE"] = "../../gurobi.lic"
const GUROBI_ENV = Gurobi.Env()

# Helper to build Adjacency Lists and filter strict zero-weight edges
function get_adj(obj)
    n, m = size(obj)
    N_S = [Int[] for _ in 1:n]
    N_R = [Int[] for _ in 1:m]
    I, J, V = findnz(obj)
    f_I, f_J = Int[], Int[]
    for (i, j, v) in zip(I, J, V)
        if v > 0
            push!(N_S[i], j)
            push!(N_R[j], i)
            push!(f_I, i)
            push!(f_J, j)
        end
    end
    return N_S, N_R, f_I, f_J
end

function alternative_sol_straight(n, m, probs, obj)
    N_S, N_R, I, J = get_adj(obj)
    
    model = Model(() -> Gurobi.Optimizer(GUROBI_ENV))
    set_attribute(model, "MemLimit", 16.0)
    set_attribute(model, "TimeLimit", 3500.0)
    set_optimizer_attribute(model, "Threads", 8)
    set_optimizer_attribute(model, "MIPGap", 0.01)
    
    # 3D Variables constrained specifically to edges where obj > 0
    triplets = Tuple{Int, Int, Int}[]
    for k in 1:m
        for i in N_R[k]
            push!(triplets, (i, n+1, k)) # j = n+1 (no backup)
            for j in N_R[k]
                if i != j
                    push!(triplets, (i, j, k))
                end
            end
        end
    end
    
    @variable(model, x[t in triplets], Bin)
    
    vars_by_person = [VariableRef[] for _ in 1:n]
    for t in triplets
        if t[1] <= n; push!(vars_by_person[t[1]], x[t]); end
        if t[2] <= n; push!(vars_by_person[t[2]], x[t]); end
    end
    for i in 1:n
        if !isempty(vars_by_person[i])
            @constraint(model, sum(vars_by_person[i]) <= 1)
        end
    end
    
    vars_by_house = [VariableRef[] for _ in 1:m]
    for t in triplets
        if t[3] <= m; push!(vars_by_house[t[3]], x[t]); end
    end
    for k in 1:m
        if !isempty(vars_by_house[k])
            @constraint(model, sum(vars_by_house[k]) <= 1)
        end
    end
    
    obj_expr = AffExpr(0.0)
    for (i, j, k) in triplets
        o_ik = obj[i, k]
        p_ik = probs[i, k]
        o_jk = (j == n+1) ? 0.0 : obj[j, k]
        coef = o_ik + o_jk * (1.0 - p_ik)
        add_to_expression!(obj_expr, coef, x[(i, j, k)])
    end
    @objective(model, Max, obj_expr)
    optimize!(model)
    
    # Build results in O(E) using triplets
    I_res = Int[]; J_res = Int[]; V_res = Float64[]
    for (i, j, k) in triplets
        if value(x[(i, j, k)]) > 0.5
            push!(I_res, i)
            push!(J_res, k)
            push!(V_res, 1.0)
        end
    end
    # Julia's sparse function automatically sums duplicate indices
    return sparse(I_res, J_res, V_res, n, m)
end

function point8_sol_straight(n, m, probs, obj)
    N_S, N_R, I, J = get_adj(obj)
    
    model = Model(() -> Gurobi.Optimizer(GUROBI_ENV))
    set_attribute(model, "MemLimit", 16.0)
    set_attribute(model, "TimeLimit", 3500.0)
    set_optimizer_attribute(model, "Threads", 8)
    set_optimizer_attribute(model, "MIPGap", 0.01)

    tuples = Tuple{Int, Int, Int, Int, Int}[]
    for k in 1:m
        for i in N_R[k]
            # Case 1: l = m+1 (dummy backup assignment house)
            for g in N_R[k]
                if g != i
                    push!(tuples, (g, i, n+1, k, m+1))
                end
            end
            push!(tuples, (n+1, i, n+1, k, m+1)) 
            
            # Case 2: l <= m (valid 2nd active house)
            for l in 1:m
                if l == k; continue; end
                for j in N_R[l]
                    if j == i; continue; end
                    for g in N_R[k]
                        if g != i && g != j
                            push!(tuples, (g, i, j, k, l))
                        end
                    end
                    for g in N_R[l]
                        if g != i && g != j && !(g in N_R[k])
                            push!(tuples, (g, i, j, k, l))
                        end
                    end
                    push!(tuples, (n+1, i, j, k, l))
                end
            end
        end
    end
    
    @variable(model, x[t in tuples], Bin)
    
    vars_by_person = [VariableRef[] for _ in 1:n]
    for t in tuples
        for idx in 1:3
            p = t[idx]
            if p <= n; push!(vars_by_person[p], x[t]); end
        end
    end
    for a in 1:n
        if !isempty(vars_by_person[a])
            @constraint(model, sum(vars_by_person[a]) <= 1)
        end
    end
    
    vars_by_house = [VariableRef[] for _ in 1:m]
    for t in tuples
        for idx in 4:5
            h = t[idx]
            if h <= m; push!(vars_by_house[h], x[t]); end
        end
    end
    for h in 1:m
        if !isempty(vars_by_house[h])
            @constraint(model, sum(vars_by_house[h]) <= 1)
        end
    end
    
    obj_expr = AffExpr(0.0)
    for t in tuples
        g, i, j, k, l = t
        o_ik = obj[i, k]
        p_ik = 1.0 - probs[i, k]
        
        o_jl = (j <= n && l <= m) ? obj[j, l] : 0.0
        p_jl = (j <= n && l <= m) ? (1.0 - probs[j, l]) : 0.0
        
        o_gk = (g <= n) ? obj[g, k] : 0.0
        o_gl = (g <= n && l <= m) ? obj[g, l] : 0.0
        
        coef = o_ik + o_jl + p_ik * o_gk + p_jl * (1.0 - p_ik) * o_gl
        add_to_expression!(obj_expr, coef, x[t])
    end
    @objective(model, Max, obj_expr)
    optimize!(model)
    
    I_res = Int[]; J_res = Int[]; V_res = Float64[]
    for t in tuples
        if value(x[t]) > 0.5
            g, i, j, k, l = t
            push!(I_res, i)
            push!(J_res, k)
            push!(V_res, 1.0)
        end
    end
    return sparse(I_res, J_res, V_res, n, m)
end

function linear(n, m, probs, obj, just_obj = true)
    N_S, N_R, I, J = get_adj(obj)
    edges = [(i, j) for (i, j) in zip(I, J)]
    
    model = Model(() -> Gurobi.Optimizer(GUROBI_ENV))
    set_attribute(model, "MemLimit", 16.0)
    set_attribute(model, "TimeLimit", 3500.0)
    set_optimizer_attribute(model, "Threads", 8)

    @variable(model, 0 <= x[e in edges] <= 1)
    @variable(model, 0 <= y[e in edges] <= 1)

    for i in 1:n
        if !isempty(N_S[i])
            @constraint(model, sum(x[(i,j)] + y[(i,j)] for j in N_S[i]) <= 1)
        end
    end
    
    for j in 1:m
        if !isempty(N_R[j])
            @constraint(model, sum(probs[i,j] * x[(i,j)] + y[(i,j)] for i in N_R[j]) <= 1)
            @constraint(model, sum(x[(i,j)] for i in N_R[j]) <= 1)
        end
    end
    
    @objective(model, Max, sum(obj[i,j] * (x[(i,j)] + y[(i,j)]) for (i,j) in edges))
    optimize!(model)
    
    if just_obj
        return objective_value(model)
    else
        Ix = Int[]; Jx = Int[]; Vx = Float64[]
        Iy = Int[]; Jy = Int[]; Vy = Float64[]
        for e in edges
            vx = value(x[e])
            vy = value(y[e])
            if vx > 1e-10; push!(Ix, e[1]); push!(Jx, e[2]); push!(Vx, vx); end
            if vy > 1e-10; push!(Iy, e[1]); push!(Jy, e[2]); push!(Vy, vy); end
        end
        return sparse(Ix, Jx, Vx, n, m), sparse(Iy, Jy, Vy, n, m)
    end
end

function one_stage_opt(n, m, probs, obj)
    N_S, N_R, I, J = get_adj(obj)
    edges = [(i, j) for (i, j) in zip(I, J)]
    
    model = Model(() -> Gurobi.Optimizer(GUROBI_ENV))
    set_attribute(model, "MemLimit", 16.0)
    set_attribute(model, "TimeLimit", 3500.0)
    set_optimizer_attribute(model, "Threads", 8)
    set_optimizer_attribute(model, "MIPGap", 0.01)
    
    @variable(model, x[e in edges], Bin)
    
    for i in 1:n
        if !isempty(N_S[i])
            @constraint(model, sum(x[(i,j)] for j in N_S[i]) <= 1)
        end
    end
    for j in 1:m
        if !isempty(N_R[j])
            @constraint(model, sum(x[(i,j)] for i in N_R[j]) <= 1)
        end
    end
    
    @objective(model, Max, sum(obj[i,j] * x[(i,j)] for (i,j) in edges))
    optimize!(model)
    
    I_res = Int[]; J_res = Int[]; V_res = Float64[]
    for e in edges
        v = value(x[e])
        if v > 0.5
            push!(I_res, e[1]); push!(J_res, e[2]); push!(V_res, 1.0)
        end
    end
    return sparse(I_res, J_res, V_res, n, m)
end

function fluid(n, m, probs, obj)
    N_S, N_R, I, J = get_adj(obj)
    edges = [(i, j) for (i, j) in zip(I, J)]
    
    model = Model(() -> Gurobi.Optimizer(GUROBI_ENV))
    set_attribute(model, "MemLimit", 16.0)
    set_attribute(model, "TimeLimit", 3500.0)
    set_optimizer_attribute(model, "MIPGap", 0.01)
    set_optimizer_attribute(model, "Threads", 8)
    
    @variable(model, x[e in edges], Bin)
    @variable(model, 0 <= y[e in edges] <= 1)

    for i in 1:n
        if !isempty(N_S[i])
            @constraint(model, sum(x[(i,j)] + y[(i,j)] for j in N_S[i]) <= 1)
        end
    end
    for j in 1:m
        if !isempty(N_R[j])
            @constraint(model, sum(probs[i,j] * x[(i,j)] + y[(i,j)] for i in N_R[j]) <= 1)
            @constraint(model, sum(x[(i,j)] for i in N_R[j]) <= 1)
        end
    end
    
    @objective(model, Max, sum(obj[i,j] * (x[(i,j)] + y[(i,j)]) for (i,j) in edges))
    optimize!(model)

    Ix = Int[]; Jx = Int[]; Vx = Float64[]
    Iy = Int[]; Jy = Int[]; Vy = Float64[]
    for e in edges
        vx = value(x[e])
        vy = value(y[e])
        if vx > 1e-10; push!(Ix, e[1]); push!(Jx, e[2]); push!(Vx, vx); end
        if vy > 1e-10; push!(Iy, e[1]); push!(Jy, e[2]); push!(Vy, vy); end
    end
    return sparse(Ix, Jx, Vx, n, m), sparse(Iy, Jy, Vy, n, m), objective_value(model)
end

function SAA_no_opt(n, m, probs, obj, s=200)
    N_S, N_R, I, J = get_adj(obj)
    edges = [(i, j) for (i, j) in zip(I, J)]
    
    # Store scenarios efficiently per edge
    scenarios = Dict{Tuple{Int,Int}, BitVector}()
    for (i, j) in edges
        scenarios[(i,j)] = rand(s) .< probs[i,j]
    end
    
    x_val = one_stage_opt(n, m, probs, obj)

    model= Model(() -> Gurobi.Optimizer(GUROBI_ENV))
    set_attribute(model, "MemLimit", 16.0)
    set_attribute(model, "TimeLimit", 3500.0)
    set_optimizer_attribute(model, "Threads", 8)
    set_optimizer_attribute(model, "MIPGap", 0.01)
    
    @variable(model, x[e in edges] , Bin)
    @variable(model, 0 <= y[e in edges, 1:s] <= 1)

    for i in 1:n
        if !isempty(N_S[i])
            for scen in 1:s
                @constraint(model, sum(x[(i, j)] + y[(i, j), scen] for j in N_S[i]) <= 1)
            end
        end
    end
    for j in 1:m
        if !isempty(N_R[j])
            @constraint(model, sum(x[(i, j)] for i in N_R[j]) <= 1)
            for scen in 1:s
                @constraint(model, 
                    sum(scenarios[(i,j)][scen] ? x[(i,j)] : 0.0 for i in N_R[j]) +
                    sum(y[(i,j), scen] for i in N_R[j]) <= 1
                )
            end
        end
    end

    obj_s = obj ./ s
    @objective(model, Max,
        sum(obj[i,j] * x[(i,j)] for (i,j) in edges) +
        sum(obj_s[i,j] * y[(i,j), scen] for (i,j) in edges, scen in 1:s)
    )

    for (i, j) in edges
        set_start_value(x[(i,j)], x_val[i,j])
    end
    optimize!(model)
    
    I_res = Int[]; J_res = Int[]; V_res = Float64[]
    for e in edges
        v = value(x[e])
        if v > 0.5
            push!(I_res, e[1]); push!(J_res, e[2]); push!(V_res, 1.0)
        end
    end
    return sparse(I_res, J_res, V_res, n, m)
end

function efficient_567(n, m, probs, obj)
    x, y = linear(n, m, probs, obj, false)
    X, _ = GKPSCompleteBipartite.gkps_round_complete(x, n, m)
    Y, _ = GKPSCompleteBipartite.gkps_round_complete(y, n, m)

    # Note: sum(..., dims=2) returns an O(n) array, completely safe.
    x_sum = vec(sum(x, dims = 2))
    y_sum = vec(sum(y, dims = 2))

    x_prime = vec(sum(X, dims = 2))
    y_prime = vec(sum(Y, dims = 2))

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

    p = 1.0 .- vec(sum(x .* probs, dims = 1)) # O(m) array
    q = rand(m) .< p
    
    I_res = Int[]; J_res = Int[]; V_res = Float64[]
    
    I_X, J_X, V_X = findnz(X)
    for (i, j, v) in zip(I_X, J_X, V_X)
        val = v * x_prime[i] * q[j]
        if abs(val) > 1e-10
            push!(I_res, i); push!(J_res, j); push!(V_res, val)
        end
    end
    
    I_Y, J_Y, V_Y = findnz(Y)
    for (i, j, v) in zip(I_Y, J_Y, V_Y)
        val = v * y_prime[i] * (1.0 - q[j])
        if abs(val) > 1e-10
            push!(I_res, i); push!(J_res, j); push!(V_res, val)
        end
    end
    
    return sparse(I_res, J_res, V_res, n, m)
end

function alt_sol_from_fluid(n, m, probs, obj, x, y)
    N_S, N_R, I_adj, J_adj = get_adj(obj)
    edges = [(i, j) for (i, j) in zip(I_adj, J_adj)]
    
    p = 1.0 .- vec(sum(x .* probs, dims = 1))
    w0 = vec(sum(x .* obj, dims = 1))

    model= Model(() -> Gurobi.Optimizer(GUROBI_ENV))
    @variable(model, 0 <= ϕ[e in edges])
    @variable(model, 0 <= ψ[e in edges])
    
    for j in 1:m
        if !isempty(N_R[j])
            @constraint(model, sum(ϕ[(i,j)] + ψ[(i,j)] for i in N_R[j]) <= 1)
        end
    end
    for i in 1:n
        if !isempty(N_S[i])
            @constraint(model, sum(ϕ[(i,j)] + ψ[(i,j)] for j in N_S[i]) <= 1)
        end
    end

    obj_expr = AffExpr(0.0)
    for (i,j) in edges
        if y[i,j] > 1e-10
            alpha = obj[i,j] - w0[j]
            beta = obj[i,j] * p[j]
        else
            alpha = -1.0
            beta = -1.0
        end
        add_to_expression!(obj_expr, alpha, ϕ[(i,j)])
        add_to_expression!(obj_expr, beta, ψ[(i,j)])
    end
    
    @objective(model, Max, obj_expr)
    optimize!(model)

    use = zeros(m)
    phi_res = Dict{Tuple{Int,Int}, Float64}()
    for (i, j) in edges
        v = value(ϕ[(i,j)])
        if v > 1e-10
            phi_res[(i, j)] = v
            use[j] += v
        end
    end
    
    I_res = Int[]; J_res = Int[]; V_res = Float64[]
    for (i, j) in edges
        v_phi = get(phi_res, (i, j), 0.0)
        final_val = v_phi + x[i, j] * (1.0 - use[j])
        if final_val > 1e-10
            push!(I_res, i); push!(J_res, j); push!(V_res, final_val)
        end
    end
    
    return sparse(I_res, J_res, V_res, n, m)
end

function get_abc(y, obj, obj_stage_1::AbstractVector{Float64}, p::AbstractVector{Float64}, tol::Float64 = 1e-9)
    nU, nV = size(y)
    N_S = [Int[] for _ in 1:nU]
    N_R = [Int[] for _ in 1:nV]
    
    I_y, J_y, V_y = findnz(y)
    for (i, j, v) in zip(I_y, J_y, V_y)
        if v > tol
            push!(N_S[i], j)
            push!(N_R[j], i)
        end
    end
    
    edges = Tuple{Int, Int}[]
    for i in 1:nU
        for k in N_S[i]; push!(edges, (i, k)); end
    end
    
    triplets = Tuple{Int, Int, Int}[]
    for i in 1:nU
        for k in N_S[i]
            for l in N_S[i]
                if k != l; push!(triplets, (i, k, l)); end
            end
        end
    end
    
    model = Model(() -> Gurobi.Optimizer(GUROBI_ENV))
    set_attribute(model, "MemLimit", 16.0)
    set_attribute(model, "TimeLimit", 3500.0)
    set_optimizer_attribute(model, "Threads", 8)
    
    @variable(model, a[e in edges] >= 0)
    @variable(model, b[e in edges] >= 0)
    @variable(model, c[t in triplets] >= 0)
    
    @objective(model, Max,
        sum((obj[i, k] - obj_stage_1[k]) * a[(i, k)] + (p[k] * obj[i, k]) * b[(i, k)] for (i, k) in edges) +
        sum((p[k] * obj[i, k] + (1 - p[k]) * p[l] * obj[i, l]) * c[(i, k, l)] for (i, k, l) in triplets)
    )
    
    for i in 1:nU
        if isempty(N_S[i]); continue; end
        @constraint(model, sum(a[(i, k)] + b[(i, k)] for k in N_S[i]) + 
                           sum(c[(i, k, l)] for k in N_S[i] for l in N_S[i] if k != l) <= 1)
    end
    
    for k in 1:nV
        if isempty(N_R[k]); continue; end
        @constraint(model, sum(a[(i, k)] + b[(i, k)] + sum(c[(i, k, l)] + c[(i, l, k)] for l in N_S[i] if l != k) for i in N_R[k]) <= 1)
    end
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        a_res = Dict(e => value(a[e]) for e in edges)
        b_res = Dict(e => value(b[e]) for e in edges)
        c_res = Dict(t => value(c[t]) for t in triplets)
        return a_res, b_res, c_res
    else
        error("Model did not solve optimally! Status: $(termination_status(model))")
    end
end

function point_8_from_fluid(n, m, p_input, obj, x, y)
    p = 1.0 .- vec(sum(x .* p_input, dims = 1))
    obj_stage_1 = vec(sum(x .* obj, dims = 1))

    a, b, c = get_abc(y, obj, obj_stage_1, p)
    
    use = zeros(m)
    for (key, val) in a
        i, k = key
        use[k] += val
    end

    I_res = Int[]; J_res = Int[]; V_res = Float64[]
    
    I_x, J_x, V_x = findnz(x)
    for (i, k, v) in zip(I_x, J_x, V_x)
        val = v * (1.0 - use[k])
        if abs(val) > 1e-10
            push!(I_res, i); push!(J_res, k); push!(V_res, val)
        end
    end
    
    for ((i, k), v) in a
        if abs(v) > 1e-10
            push!(I_res, i); push!(J_res, k); push!(V_res, v)
        end
    end

    # sparse() gracefully sums items with overlapping (i, k) coordinates
    return sparse(I_res, J_res, V_res, n, m)
end

function get_value(sol, scenarios::Dict{Tuple{Int,Int}, BitVector}, n, m, probs, obj)
    s = length(first(values(scenarios)))
    N_S, N_R, I, J = get_adj(obj)
    edges = [(i, j) for (i, j) in zip(I, J)]

    model = Model(() -> Gurobi.Optimizer(GUROBI_ENV))
    set_silent(model)
    @variable(model, 0 <= y[e in edges] <= 1)

    sol_sums = vec(sum(sol, dims=2))
    for i in 1:n
        if !isempty(N_S[i])
            @constraint(model, sum(y[(i,j)] for j in N_S[i]) <= 1.0 - sol_sums[i])
        end
    end

    houses_con = Dict{Int, ConstraintRef}()
    for j in 1:m
        if !isempty(N_R[j])
            houses_con[j] = @constraint(model, sum(y[(i,j)] for i in N_R[j]) <= 1.0)
        end
    end

    @objective(model, Max, sum(obj[i,j] * y[(i,j)] for (i,j) in edges))

    v = 0.0
    I_s, J_s, V_s = findnz(sol)
    for scen in 1:s
        contrib = zeros(m)
        for (i, j, val) in zip(I_s, J_s, V_s)
            if get(scenarios, (i,j), falses(s))[scen]
                contrib[j] += val
            end
        end
        for j in 1:m
            if haskey(houses_con, j)
                set_normalized_rhs(houses_con[j], 1.0 - contrib[j])
            end
        end
        optimize!(model)
        v += objective_value(model)
    end
    return v / s + sum(sol .* obj)
end

function generate(m)
    n = Int(floor(m * 1.5))

    I = Int[]; J = Int[]; O = Float64[]; P = Float64[]

    # high value
    for i=1:n
        ms = 1 .+ Int.(floor.(rand(2) .* 1.5 .* m))
        for k in ms
            if k<= m
                push!(I, i); push!(J, k); push!(O, 1.0); push!(P, 1.0)
            end
        end
    end

    # high efficiency
    for i=1:n
        ms = 1 .+ Int.(floor.(rand(2) .* 1.5 .* m))
        for k in ms
            if k<= m
                prob = rand() / 2
                push!(I, i); push!(J, k); push!(O, min(0.99, (3+rand()) * prob)); push!(P, prob)
            end
        end
    end

    # Discard non-overwriting duplicates to build standard sparse matrix representations explicitly
    seen = Set{Tuple{Int,Int}}()
    f_I = Int[]; f_J = Int[]; f_O = Float64[]; f_P = Float64[]
    for idx in length(I):-1:1
        if !((I[idx], J[idx]) in seen)
            push!(seen, (I[idx], J[idx]))
            push!(f_I, I[idx]); push!(f_J, J[idx]); push!(f_O, O[idx]); push!(f_P, P[idx])
        end
    end

    obj = sparse(f_I, f_J, f_O, n, m)
    probs = sparse(f_I, f_J, f_P, n, m)
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

    I_o, J_o, V_o = findnz(obj)
    df = DataFrame(i=I_o, j=J_o, obj=V_o)
    CSV.write("$dir/obj_$m.csv", df)
    
    I_p, J_p, V_p = findnz(probs)
    df = DataFrame(i=I_p, j=J_p, probs=V_p)
    CSV.write("$dir/probs_$m.csv", df)

    x, y, val = fluid_data.value
    
    I_x, J_x, V_x = findnz(x)
    df = DataFrame(i=I_x, j=J_x, x=V_x)
    CSV.write("$dir/fluid_x_$m.csv", df)
    
    I_y, J_y, V_y = findnz(y)
    df = DataFrame(i=I_y, j=J_y, y=V_y)
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

function write_sol(sol::SparseMatrixCSC, filename::String)
    I, J, V = findnz(sol)
    CSV.write(filename, DataFrame(i=I, j=J, val=V))
end

function read_sol(filename::String, n::Int, m::Int)
    df = CSV.read(filename, DataFrame)
    return sparse(df.i, df.j, df.val, n, m)
end

function main(m, index)
    dir = "results_$m"

    if index == 0
        initialize(m)
        
    elseif index <= length(functions)
        n = Int(floor(m * 1.5))

        # Reconstruct sparse matrices from triplet CSVs
        df_obj = CSV.read("$dir/obj_$m.csv", DataFrame)
        obj = sparse(df_obj.i, df_obj.j, df_obj.obj, n, m)
        
        df_probs = CSV.read("$dir/probs_$m.csv", DataFrame)
        probs = sparse(df_probs.i, df_probs.j, df_probs.probs, n, m)
        
        if functions[index]["requires_fluid"]
            df_x = CSV.read("$dir/fluid_x_$m.csv", DataFrame)
            x = sparse(df_x.i, df_x.j, df_x.x, n, m)
            
            df_y = CSV.read("$dir/fluid_y_$m.csv", DataFrame)
            y = sparse(df_y.i, df_y.j, df_y.y, n, m)
            
            ft = 0.0
            open("$dir/results_$m.txt", "r") do io
                ft += parse(Float64, readline(io))
            end
            
            function_data = @timed functions[index]["function"](n, m, probs, obj, x, y)

            touch("$dir/$(functions[index]["name"])_$m.txt")
            open("$dir/$(functions[index]["name"])_$m.txt", "w") do io
                write(io, "$(function_data.time + ft)\n")
            end
            
            # Using updated write_sol (saved as CSV for ease of sparse reloading)
            write_sol(function_data.value, "$dir/$(functions[index]["name"])_sol_$m.csv")
        else
            function_data = @timed functions[index]["function"](n, m, probs, obj)

            touch("$dir/$(functions[index]["name"])_$m.txt")
            open("$dir/$(functions[index]["name"])_$m.txt", "w") do io
                write(io, "$(function_data.time)\n")
            end
            
            write_sol(function_data.value, "$dir/$(functions[index]["name"])_sol_$m.csv")
        end
        
    elseif index == length(functions) + 1
        n = Int(floor(m * 1.5))

        df_obj = CSV.read("$dir/obj_$m.csv", DataFrame)
        obj = sparse(df_obj.i, df_obj.j, df_obj.obj, n, m)
        
        df_probs = CSV.read("$dir/probs_$m.csv", DataFrame)
        probs = sparse(df_probs.i, df_probs.j, df_probs.probs, n, m)

        # Generate Scenarios sparsely using the explicit Dictionary structure
        s = 200
        scenarios = Dict{Tuple{Int,Int}, BitVector}()
        I_p, J_p, V_p = findnz(probs)
        for (i, j, p) in zip(I_p, J_p, V_p)
            scenarios[(i,j)] = rand(s) .< p
        end
        
        for f in functions
            sol_file = "$dir/$(f["name"])_sol_$m.csv"
            if isfile(sol_file)
                x_sol = read_sol(sol_file, n, m)
                val = get_value(x_sol, scenarios, n, m, probs, obj)
                
                open("$dir/$(f["name"])_$m.txt", "a") do io
                    println(io, "$(val)")
                end
            end
        end
        
        for f in functions
            sol_file = "$dir/$(f["name"])_sol_$m.csv"
            if isfile(sol_file)
                open("$dir/$(f["name"])_$m.txt", "r") do io
                    t = parse(Float64, readline(io))
                    v = parse(Float64, readline(io))
                    
                    open("$dir/results_$m.txt", "a") do io_res
                        println(io_res, "$(f["name"]) val: $v")
                        println(io_res, "$(f["name"]) time: $t")
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

# for i=0:8
#     println(i, "\n\n\n\n\n")
#     main(10, i)
# end