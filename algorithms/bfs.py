from collections import deque

def bfs(graph, start, goals):
    queue = deque([(start, [start])])  # (node, path to node)
    visited = set()
    number_of_nodes = 0

    while queue:
        current, path = queue.popleft()
        number_of_nodes += 1

        if current in goals:
            return current, number_of_nodes, path

        if current not in visited:
            visited.add(current)
            for neighbor, _ in sorted(graph.get(current, [])):
                if neighbor not in visited and neighbor not in [n for n, _ in queue]:
                    queue.append((neighbor, path + [neighbor]))

    return None, number_of_nodes, []
