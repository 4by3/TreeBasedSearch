def dfs(graph, start, goals):
    stack = [(0, start, [start])]
    visited = set()
    number_of_nodes = 0

    while stack:
        cost, node, path = stack.pop()

        if node in visited:
            continue
        visited.add(node)
        number_of_nodes += 1

        if node in goals:
            return node, number_of_nodes, path, cost

        if node in graph:
            for neighbor, node_cost in sorted(graph[node], reverse=True):
                if neighbor not in visited:
                    stack.append((cost + node_cost, neighbor, path + [neighbor]))

    return None, number_of_nodes, []  # No valid path found