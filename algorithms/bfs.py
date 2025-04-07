from collections import deque

def bfs(graph, start, goal):
    queue = deque([(start, [start])])  # Queue stores (current node, path to reach it)
    visited = set()
    nodes_explored = 0

    while queue:
        current, path = queue.popleft()
        nodes_explored += 1

        if current == goal:
            return path, nodes_explored

        if current not in visited:
            visited.add(current)
            for neighbor, _ in sorted(graph.get(current, [])):  # Expand in ascending order
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
    
    return None, nodes_explored  # No path found

# Example usage
if __name__ == "__main__":
    sample_graph = {
        1: [(2, 1), (3, 4)],
        2: [(1, 1), (4, 2)],
        3: [(1, 4), (4, 3)],
        4: [(2, 2), (3, 3)]
    }
    print(bfs(sample_graph, 1, 4))
