module GKPSCompleteBipartite

export gkps_round_complete

using Random
using SparseArrays

"""
    gkps_round_complete(X::SparseMatrixCSC, nU, nV; rng=Random.default_rng(), eps=1e-12, check=true)

GKPS dependent rounding for a fractional bipartite matching.

Input:
- X: A SparseMatrixCSC representing the fractional assignment.
- Assumes 0 <= X[i,j] <= 1 and row/col sums <= 1 (if check=true).

Output:
- X_round::SparseMatrixCSC rounded to {0,1} (up to eps)
- chosen_edges::Vector{Tuple{Int,Int}} pairs (i,j) with X_round edge = 1
"""
function gkps_round_complete(X::SparseMatrixCSC{<:Real, <:Integer}, nU::Int, nV::Int;
        rng::AbstractRNG = Random.default_rng(),
        eps::Float64 = 1e-12,
        check::Bool = true)

    I, J, V = findnz(X)
    m = length(V)
    x = Vector{Float64}(undef, m)
    
    @inbounds for e in 1:m
        xe = Float64(V[e])
        if xe ≤ eps
            xe = 0.0
        elseif xe ≥ 1.0 - eps
            xe = 1.0
        end
        x[e] = clamp(xe, 0.0, 1.0)
    end

    if check
        ok, msg = check_feasible_sparse(x, I, J, nU, nV; tol=1e-9)
        ok || error(msg)
    end

    # 1. PREALLOCATE AND BUILD GRAPH EXACTLY ONCE
    adjU = [Int[] for _ in 1:nU]
    adjV = [Int[] for _ in 1:nV]
    degU = zeros(Int, nU)
    degV = zeros(Int, nV)
    
    active_frac = 0
    @inbounds for e in 1:m
        if x[e] > eps && x[e] < 1.0 - eps
            push!(adjU[I[e]], e)
            push!(adjV[J[e]], e)
            degU[I[e]] += 1
            degV[J[e]] += 1
            active_frac += 1
        end
    end

    # 2. PREALLOCATE WORKSPACE ARRAYS (Zero inner-loop allocations)
    N = nU + nV
    state = zeros(Int, N)
    parentV = zeros(Int, N)
    parentE = zeros(Int, N)
    visited = falses(N)
    
    stack = Tuple{Int,Int}[]
    sizehint!(stack, N)
    q = Int[]
    sizehint!(q, N)
    visited_nodes = Int[]
    sizehint!(visited_nodes, N)

    # 3. FAST IN-PLACE LOOP
    while active_frac > 0
        Eseq = find_cycle_edges(adjU, adjV, degU, degV, nU, nV, I, J, x, eps, state, parentV, parentE, stack, visited_nodes)
        
        # O(|V_visited|) cleanup instead of O(|V|)
        @inbounds for v in visited_nodes
            state[v] = 0
        end

        if Eseq === nothing
            Eseq = find_leaf_to_leaf_path(adjU, adjV, degU, degV, nU, nV, I, J, x, eps, visited, parentV, parentE, q, visited_nodes)
            @inbounds for v in visited_nodes
                visited[v] = false
            end
        end
        @assert Eseq !== nothing

        if length(Eseq) == 1
            e = Eseq[1]
            x[e] = (rand(rng) < x[e]) ? 1.0 : 0.0
            degU[I[e]] -= 1
            degV[J[e]] -= 1
            active_frac -= 1
        else
            gkps_update!(x, Eseq; rng=rng, eps=eps)
            # Update degrees incrementally ONLY for edges that just got fixed
            for e in Eseq
                if x[e] ≤ eps || x[e] ≥ 1.0 - eps
                    degU[I[e]] -= 1
                    degV[J[e]] -= 1
                    active_frac -= 1
                end
            end
        end
    end

    # Finally, construct result cleanly
    I_res = Int[]; J_res = Int[]; V_res = Float64[]
    chosen = Tuple{Int,Int}[]
    
    @inbounds for e in 1:m
        if x[e] ≥ 1.0 - eps
            x[e] = 1.0
            push!(I_res, I[e])
            push!(J_res, J[e])
            push!(V_res, 1.0)
            push!(chosen, (I[e], J[e]))
        else
            x[e] = 0.0
        end
    end
    
    return sparse(I_res, J_res, V_res, nU, nV), chosen
end


# ------------------------- Feasibility check -------------------------

function check_feasible_sparse(x::Vector{Float64}, I::Vector{Int}, J::Vector{Int}, nU::Int, nV::Int; tol::Float64=1e-9)
    m = length(x)
    rowsum = zeros(Float64, nU)
    colsum = zeros(Float64, nV)

    @inbounds for e in 1:m
        xe = x[e]
        if xe < -tol || xe > 1.0 + tol
            return (false, "x[$e] = $xe not in [0,1] (tol=$tol)")
        end
        rowsum[I[e]] += xe
        colsum[J[e]] += xe
    end
    @inbounds for i in 1:nU
        if rowsum[i] > 1.0 + tol
            return (false, "row $i sum = $(rowsum[i]) exceeds 1 (tol=$tol)")
        end
    end
    @inbounds for j in 1:nV
        if colsum[j] > 1.0 + tol
            return (false, "col $j sum = $(colsum[j]) exceeds 1 (tol=$tol)")
        end
    end
    return (true, "")
end


# ------------------------- GKPS alternating update -------------------------

function gkps_update!(x::Vector{Float64}, Eseq::Vector{Int}; rng::AbstractRNG, eps::Float64)
    k = length(Eseq)
    @assert k ≥ 2

    α = Inf
    β = Inf

    @inbounds for t in 1:k
        e = Eseq[t]
        s = isodd(t) ? +1.0 : -1.0
        xe = x[e]
        α = min(α, (s > 0) ? (1.0 - xe) : xe)      # for direction A
        β = min(β, (s > 0) ? xe : (1.0 - xe))      # for direction B
    end

    α = max(0.0, α)
    β = max(0.0, β)
    if α + β ≤ eps
        return nothing
    end

    pA = β / (α + β)
    if rand(rng) < pA
        @inbounds for t in 1:k
            e = Eseq[t]
            s = isodd(t) ? +1.0 : -1.0
            x[e] = clamp(x[e] + s*α, 0.0, 1.0)
        end
    else
        @inbounds for t in 1:k
            e = Eseq[t]
            s = isodd(t) ? +1.0 : -1.0
            x[e] = clamp(x[e] - s*β, 0.0, 1.0)
        end
    end
    return nothing
end


# ------------------------- Graph helpers (unified vertex IDs) -------------------------

@inline function other_vertex(e::Int, curr::Int, nU::Int, I::Vector{Int}, J::Vector{Int})
    uvid = I[e]
    vvid = nU + J[e]
    return (curr == uvid) ? vvid : uvid
end

@inline function incident_edges(vid::Int, adjU, adjV, nU::Int)
    vid ≤ nU ? adjU[vid] : adjV[vid - nU]
end


# ------------------------- Cycle finding (DFS) -------------------------
function find_cycle_edges(adjU, adjV, degU, degV, nU::Int, nV::Int, I, J, x, eps, state, parentV, parentE, stack, visited_nodes)
    empty!(stack)
    empty!(visited_nodes)
    N = nU + nV
    
    for s in 1:N
        d = (s ≤ nU) ? degU[s] : degV[s - nU]
        if d == 0 || state[s] != 0
            continue
        end

        push!(stack, (s, 1))
        state[s] = 1
        parentV[s] = 0
        parentE[s] = 0
        push!(visited_nodes, s)

        while !isempty(stack)
            v, idx = stack[end]
            inc = incident_edges(v, adjU, adjV, nU)
            
            valid_edge = false
            local e, w
            while idx ≤ length(inc)
                e = inc[idx]
                idx += 1
                
                # Lazy skip of mathematically fixed edges
                if x[e] > eps && x[e] < 1.0 - eps
                    w = other_vertex(e, v, nU, I, J)
                    if w != parentV[v]
                        valid_edge = true
                        break
                    end
                end
            end
            
            stack[end] = (v, idx) # save updated iterator

            if !valid_edge
                state[v] = 2
                pop!(stack)
                continue
            end

            if state[w] == 0
                state[w] = 1
                parentV[w] = v
                parentE[w] = e
                push!(stack, (w, 1))
                push!(visited_nodes, w)
            elseif state[w] == 1
                edges = Int[]
                cur = v
                while cur != w
                    push!(edges, parentE[cur])
                    cur = parentV[cur]
                end
                reverse!(edges)
                push!(edges, e)
                return edges
            end
        end
    end
    return nothing
end

function find_leaf_to_leaf_path(adjU, adjV, degU, degV, nU::Int, nV::Int, I, J, x, eps, visited, parentV, parentE, q, visited_nodes)
    empty!(visited_nodes)
    N = nU + nV
    start = 0
    @inbounds for v in 1:N
        d = (v ≤ nU) ? degU[v] : degV[v - nU]
        if d == 1
            start = v
            break
        end
    end
    @assert start != 0 "No leaf found."

    empty!(q)
    push!(q, start)
    visited[start] = true
    push!(visited_nodes, start)
    
    target = 0
    head = 1
    while head ≤ length(q) && target == 0
        v = q[head]; head += 1
        for e in incident_edges(v, adjU, adjV, nU)
            if x[e] > eps && x[e] < 1.0 - eps
                w = other_vertex(e, v, nU, I, J)
                if !visited[w]
                    visited[w] = true
                    parentV[w] = v
                    parentE[w] = e
                    push!(q, w)
                    push!(visited_nodes, w)
                    
                    dw = (w ≤ nU) ? degU[w] : degV[w - nU]
                    if dw == 1 && w != start
                        target = w
                        break
                    end
                end
            end
        end
    end

    edges = Int[]
    cur = target
    while cur != start
        push!(edges, parentE[cur])
        cur = parentV[cur]
    end
    reverse!(edges)
    return edges
end

end # module