
module MyTree

export TreeNode, build_forest

struct TreeNode
    index::Int
    is_row::Bool
    weight_to_parent::Float64
    children::Vector{TreeNode}
end

"""
    build_forest(X_frac::AbstractMatrix{Float64}; tol=1e-9)

Takes an nU x nV bipartite adjacency matrix representing a fractional matching.
Returns a `Vector{TreeNode}` representing the roots of each disconnected tree component.
Throws an error if a cycle is detected.
"""
function build_forest(X_frac::AbstractMatrix{Float64}; tol::Float64=1e-9)
    nU, nV = size(X_frac)
    N = nU + nV
    
    # Build a unified adjacency list: 
    # Rows are 1:nU, Cols are (nU+1):(nU+nV)
    adj = [Tuple{Int, Float64}[] for _ in 1:N]
    for j in 1:nV
        for i in 1:nU
            w = X_frac[i, j]
            if w > tol
                push!(adj[i], (nU + j, w))
                push!(adj[nU + j], (i, w))
            end
        end
    end
    
    visited = falses(N)
    forest = TreeNode[]
    
    # DFS helper function to build the tree recursively
    function dfs(curr_id::Int, parent_id::Int, weight_to_curr::Float64)
        visited[curr_id] = true
        
        # Determine if current node is a Row (U) or Col (V)
        is_row = curr_id <= nU
        idx = is_row ? curr_id : curr_id - nU
        
        node = TreeNode(idx, is_row, weight_to_curr, TreeNode[])
        
        for (neighbor_id, w) in adj[curr_id]
            if neighbor_id != parent_id
                if visited[neighbor_id]
                    error("Cycle detected at $(is_row ? "Row" : "Col") $idx ! The non-zero edges do not strictly form a forest.")
                end
                
                child_node = dfs(neighbor_id, curr_id, w)
                push!(node.children, child_node)
            end
        end
        
        return node
    end
    
    # Iterate through all vertices to find unvisited tree roots
    for i in 1:N
        # Only start a new tree if the node has edges and hasn't been visited
        if !visited[i] && !isempty(adj[i])
            root_node = dfs(i, 0, 0.0)
            push!(forest, root_node)
        end
    end
    
    return forest
end



end # module