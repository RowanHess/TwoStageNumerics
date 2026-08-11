module GKPSCompleteBipartite

export gkps_round_complete

using Random
using SparseArrays

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

    N = nU + nV

    # --- Preallocate Flat Adjacency List Workspace ---
    head = zeros(Int, N)
    deg = zeros(Int, N)
    
    next_arc = zeros(Int, m * 2)
    to = zeros(Int, m * 2)
    edge_id = zeros(Int, m * 2)
    
    active_edges = Int[]
    sizehint!(active_edges, m)
    @inbounds for e in 1:m
        if x[e] > eps && x[e] < 1.0 - eps
            push!(active_edges, e)
        end
    end
    
    active_nodes = Int[]
    sizehint!(active_nodes, N)
    
    # --- Preallocate DFS/BFS Workspace ---
    state = zeros(Int, N)
    parentV = zeros(Int, N)
    parentE = zeros(Int, N)
    visited = falses(N)
    
    stack = Tuple{Int,Int}[]
    sizehint!(stack, N)
    q = Int[]
    sizehint!(q, N)
    visited_states = Int[]
    sizehint!(visited_states, N)

    # --- Main GKPS Loop ---
    while !isempty(active_edges)
        
        # 1. In-place Graph Build & Active Edge Filtering (O(E_active))
        arc_idx = 1
        keep_idx = 1
        @inbounds for i in 1:length(active_edges)
            e = active_edges[i]
            if x[e] > eps && x[e] < 1.0 - eps
                active_edges[keep_idx] = e
                keep_idx += 1
                
                u = I[e]
                v = nU + J[e]
                
                to[arc_idx] = v
                edge_id[arc_idx] = e
                next_arc[arc_idx] = head[u]
                head[u] = arc_idx
                arc_idx += 1
                
                to[arc_idx] = u
                edge_id[arc_idx] = e
                next_arc[arc_idx] = head[v]
                head[v] = arc_idx
                arc_idx += 1
                
                if deg[u] == 0 push!(active_nodes, u) end
                deg[u] += 1
                
                if deg[v] == 0 push!(active_nodes, v) end
                deg[v] += 1
            end
        end
        resize!(active_edges, keep_idx - 1)
        
        isempty(active_edges) && break
        
        # 2. Find Cycle or Leaf-to-Leaf Path
        Eseq = find_cycle_edges(head, next_arc, to, edge_id, state, parentV, parentE, stack, active_nodes, visited_states)
        
        if Eseq === nothing
            Eseq = find_leaf_to_leaf_path(head, next_arc, to, edge_id, deg, visited, parentV, parentE, q, active_nodes, visited_states)
        end
        
        # 3. Update Fractional Variables
        if length(Eseq) == 1
            e = Eseq[1]
            x[e] = (rand(rng) < x[e]) ? 1.0 : 0.0
        else
            gkps_update!(x, Eseq; rng=rng, eps=eps)
        end
        
        # 4. Clean Graph Trackers Strictly for Touched Nodes (O(V_active))
        @inbounds for u in active_nodes
            head[u] = 0
            deg[u] = 0
        end
        empty!(active_nodes)
    end

    # Build Final Sparse Matrix
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

# ------------------------- Search Functions -------------------------

function find_cycle_edges(head::Vector{Int}, next_arc::Vector{Int}, to::Vector{Int}, edge_id::Vector{Int},
                          state::Vector{Int}, parentV::Vector{Int}, parentE::Vector{Int}, 
                          stack::Vector{Tuple{Int,Int}}, active_nodes::Vector{Int}, visited_states::Vector{Int})
    empty!(stack)
    empty!(visited_states)
    
    @inbounds for s in active_nodes
        if state[s] != 0
            continue
        end
        
        push!(stack, (s, head[s]))
        state[s] = 1
        parentV[s] = 0
        parentE[s] = 0
        push!(visited_states, s)
        
        while !isempty(stack)
            v, arc = stack[end]
            
            e_val::Int = 0
            w_val::Int = 0
            valid_edge::Bool = false
            
            while arc != 0
                w = to[arc]
                e = edge_id[arc]
                n_arc = next_arc[arc]
                
                if w != parentV[v]
                    e_val = e
                    w_val = w
                    valid_edge = true
                    stack[end] = (v, n_arc)
                    break
                end
                arc = n_arc
            end
            
            if !valid_edge
                state[v] = 2
                pop!(stack)
                continue
            end
            
            w = w_val
            e = e_val
            
            if state[w] == 0
                state[w] = 1
                parentV[w] = v
                parentE[w] = e
                push!(stack, (w, head[w]))
                push!(visited_states, w)
            elseif state[w] == 1
                # Cycle found!
                edges = Int[]
                cur = v
                while cur != w
                    push!(edges, parentE[cur])
                    cur = parentV[cur]
                end
                reverse!(edges)
                push!(edges, e)
                
                for node in visited_states
                    state[node] = 0
                end
                return edges
            end
        end
    end
    
    @inbounds for node in visited_states
        state[node] = 0
    end
    return nothing
end

function find_leaf_to_leaf_path(head::Vector{Int}, next_arc::Vector{Int}, to::Vector{Int}, edge_id::Vector{Int},
                                deg::Vector{Int}, visited::BitVector, parentV::Vector{Int}, parentE::Vector{Int}, 
                                q::Vector{Int}, active_nodes::Vector{Int}, visited_states::Vector{Int})
    empty!(visited_states)
    
    start = 0
    @inbounds for v in active_nodes
        if deg[v] == 1
            start = v
            break
        end
    end
    
    empty!(q)
    push!(q, start)
    visited[start] = true
    push!(visited_states, start)
    
    target = 0
    head_q = 1
    @inbounds while head_q ≤ length(q) && target == 0
        v = q[head_q]
        head_q += 1
        
        arc = head[v]
        while arc != 0
            w = to[arc]
            e = edge_id[arc]
            
            if !visited[w]
                visited[w] = true
                parentV[w] = v
                parentE[w] = e
                push!(q, w)
                push!(visited_states, w)
                
                if deg[w] == 1 && w != start
                    target = w
                    break
                end
            end
            arc = next_arc[arc]
        end
    end
    
    edges = Int[]
    cur = target
    @inbounds while cur != start
        push!(edges, parentE[cur])
        cur = parentV[cur]
    end
    reverse!(edges)
    
    @inbounds for node in visited_states
        visited[node] = false
    end
    
    return edges
end

# ------------------------- Updates / Sanity Checks -------------------------

function gkps_update!(x::Vector{Float64}, Eseq::Vector{Int}; rng::AbstractRNG, eps::Float64)
    k = length(Eseq)
    α = Inf
    β = Inf
    
    @inbounds for t in 1:k
        e = Eseq[t]
        s = isodd(t) ? +1.0 : -1.0
        xe = x[e]
        α = min(α, (s > 0) ? (1.0 - xe) : xe)
        β = min(β, (s > 0) ? xe : (1.0 - xe))
    end

    α = max(0.0, α)
    β = max(0.0, β)
    if α + β ≤ eps return end

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
end

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
        if rowsum[i] > 1.0 + tol return (false, "row $i sum exceeds 1") end
    end
    @inbounds for j in 1:nV
        if colsum[j] > 1.0 + tol return (false, "col $j sum exceeds 1") end
    end
    return (true, "")
end

end # module