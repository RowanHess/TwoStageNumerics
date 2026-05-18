module GKPSCompleteBipartite

export gkps_round_complete

using Random

"""
    gkps_round_complete(xvec, nU, nV; rng=Random.default_rng(), eps=1e-12, check=true)

GKPS dependent rounding for a fractional matching on the complete bipartite graph K_{nU,nV}.

Input:
- xvec length nU*nV, interpreted as X = reshape(xvec, nU, nV) in Julia column-major order.
  Edge (i,j) is at linear index e = i + (j-1)*nU.
- Assumes 0<=X[i,j]<=1 and row/col sums <= 1 (if check=true).

Output:
- xround::Vector{Float64} rounded to {0,1} (up to eps), same layout as xvec
- chosen_edges::Vector{Tuple{Int,Int}} pairs (i,j) with xround edge = 1
"""
function gkps_round_complete(xvec::AbstractVector{<:Real}, nU::Int, nV::Int;
        rng::AbstractRNG = Random.default_rng(),
        eps::Float64 = 1e-12,
        check::Bool = true)

    m = nU * nV
    @assert length(xvec) == m "length(xvec) must be nU*nV"

    x = Vector{Float64}(undef, m)
    # FIX: Clean numerical dust initially to prevent saturating nodes from breaking constraints
    @inbounds for e in 1:m
        xe = Float64(xvec[e])
        if xe ≤ eps
            xe = 0.0
        elseif xe ≥ 1.0 - eps
            xe = 1.0
        end
        x[e] = clamp(xe, 0.0, 1.0)
    end

    if check
        ok, msg = check_feasible_complete(x, nU, nV; tol=1e-9)
        ok || error(msg)
    end

    while true
        frac_edges = Int[]
        @inbounds for e in 1:m
            if (x[e] > eps) && (x[e] < 1.0 - eps)
                push!(frac_edges, e)
            end
        end
        isempty(frac_edges) && break

        # Build adjacency of fractional support graph H
        adjU = [Int[] for _ in 1:nU]
        adjV = [Int[] for _ in 1:nV]
        degU = zeros(Int, nU)
        degV = zeros(Int, nV)

        @inbounds for e in frac_edges
            i, j = edge_endpoints(e, nU)
            push!(adjU[i], e)
            push!(adjV[j], e)
            degU[i] += 1
            degV[j] += 1
        end

        # Prefer an even cycle if one exists; otherwise take a leaf-to-leaf path
        Eseq = find_cycle_edges(adjU, adjV, degU, degV, nU, nV)
        if Eseq === nothing
            Eseq = find_leaf_to_leaf_path(adjU, adjV, degU, degV, nU, nV)
        end
        @assert Eseq !== nothing

        # Single-edge component: Bernoulli rounding
        if length(Eseq) == 1
            e = Eseq[1]
            x[e] = (rand(rng) < x[e]) ? 1.0 : 0.0
            continue
        end

        # Otherwise do GKPS alternating update
        gkps_update!(x, Eseq; rng=rng, eps=eps)
    end

    # Snap the final structure cleanly to exact {0,1}
    @inbounds for e in 1:m
        if x[e] ≤ eps
            x[e] = 0.0
        elseif x[e] ≥ 1.0 - eps
            x[e] = 1.0
        end
    end

    chosen = Tuple{Int,Int}[]
    @inbounds for e in 1:m
        if x[e] == 1.0
            i, j = edge_endpoints(e, nU)
            push!(chosen, (i, j))
        end
    end
    return x, chosen
end


# ------------------------- Feasibility check -------------------------

function check_feasible_complete(x::Vector{Float64}, nU::Int, nV::Int; tol::Float64=1e-9)
    m = nU*nV
    length(x) == m || return (false, "x has wrong length")

    rowsum = zeros(Float64, nU)
    colsum = zeros(Float64, nV)

    @inbounds for e in 1:m
        xe = x[e]
        if xe < -tol || xe > 1.0 + tol
            return (false, "x[$e] = $xe not in [0,1] (tol=$tol)")
        end
        i, j = edge_endpoints(e, nU)
        rowsum[i] += xe
        colsum[j] += xe
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

@inline function edge_endpoints(e::Int, nU::Int)
    j = (e - 1) ÷ nU + 1
    i = (e - 1) % nU + 1
    return i, j
end

@inline function other_vertex(e::Int, curr::Int, nU::Int)
    i, j = edge_endpoints(e, nU)
    uvid = i
    vvid = nU + j
    return (curr == uvid) ? vvid : uvid
end

@inline function incident_edges(vid::Int, adjU, adjV, nU::Int)
    vid ≤ nU ? adjU[vid] : adjV[vid - nU]
end


# ------------------------- Cycle finding (DFS) -------------------------

function find_cycle_edges(adjU, adjV, degU, degV, nU::Int, nV::Int)
    N = nU + nV
    deg = zeros(Int, N)
    @inbounds for i in 1:nU; deg[i] = degU[i]; end
    @inbounds for j in 1:nV; deg[nU+j] = degV[j]; end

    state   = zeros(Int, N)     # 0 unvisited, 1 in stack, 2 done
    parentV = zeros(Int, N)
    parentE = zeros(Int, N)

    for s in 1:N
        deg[s] == 0 && continue
        state[s] != 0 && continue

        stack = Tuple{Int,Int}[]   # (vertex, next-neighbor-index)
        push!(stack, (s, 1))
        state[s] = 1
        parentV[s] = 0
        parentE[s] = 0

        while !isempty(stack)
            v, idx = stack[end]
            inc = incident_edges(v, adjU, adjV, nU)
            if idx > length(inc)
                state[v] = 2
                pop!(stack)
                continue
            end

            # advance iterator position
            stack[end] = (v, idx + 1)

            e = inc[idx]
            w = other_vertex(e, v, nU)

            if state[w] == 0
                state[w] = 1
                parentV[w] = v
                parentE[w] = e
                push!(stack, (w, 1))
            elseif state[w] == 1 && parentV[v] != w
                # Found back-edge to an ancestor in current DFS stack => cycle
                edges = Int[]
                cur = v
                while cur != w
                    pe = parentE[cur]
                    push!(edges, pe)
                    cur = parentV[cur]
                end
                reverse!(edges)         # now edges go from w to v
                push!(edges, e)         # close cycle (v,w)
                return edges
            end
        end
    end
    return nothing
end


# ------------------------- Leaf-to-leaf path in a forest (BFS) -------------------------

function find_leaf_to_leaf_path(adjU, adjV, degU, degV, nU::Int, nV::Int)
    N = nU + nV
    deg = zeros(Int, N)
    @inbounds for i in 1:nU; deg[i] = degU[i]; end
    @inbounds for j in 1:nV; deg[nU+j] = degV[j]; end

    start = 0
    @inbounds for v in 1:N
        if deg[v] == 1
            start = v
            break
        end
    end
    @assert start != 0 "No leaf found; this should not happen if support is acyclic and nonempty."

    parentV = fill(0, N)
    parentE = fill(0, N)
    visited = falses(N)

    q = Int[]
    push!(q, start)
    visited[start] = true

    target = 0
    head = 1
    while head ≤ length(q) && target == 0
        v = q[head]; head += 1
        for e in incident_edges(v, adjU, adjV, nU)
            w = other_vertex(e, v, nU)
            if !visited[w]
                visited[w] = true
                parentV[w] = v
                parentE[w] = e
                push!(q, w)
                if deg[w] == 1 && w != start
                    target = w
                    break
                end
            end
        end
    end
    @assert target != 0 "Failed to find a second leaf (unexpected in a nontrivial forest component)."

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