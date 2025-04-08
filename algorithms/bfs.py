from collections import deque

def bfs(graph, start, goals):
    queue = deque([(0, start, [start])])  # (node, path to node)
    visited = set()
    number_of_nodes = 0

    while queue:
        cost, node, path = queue.popleft()

        if node in visited:
            continue
        visited.add(node)
        number_of_nodes += 1


        if node in goals:
            return node, number_of_nodes, path, cost

        if node in graph:
            for neighbor, node_cost in sorted(graph[node]):
                if neighbor not in visited:
                    queue.append((cost + node_cost, neighbor, path + [neighbor]))

    return None, number_of_nodes, [], None