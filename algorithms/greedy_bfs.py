import heapq
from algorithms.heuristic_euclidean import heuristic_euclidean

def greedy_bfs(graph, nodes, start, goals):
    priority_queue = []
    heapq.heappush(priority_queue, (heuristic_euclidean(start, nodes, goals), start, [start]))

    visited = set()
    number_of_nodes = 0

    while priority_queue:
        _, current_node, path = heapq.heappop(priority_queue)

        if current_node in visited:
            continue
        visited.add(current_node)
        number_of_nodes += 1

        if current_node in goals:
            return current_node, number_of_nodes, path

        for neighbor, _ in graph.get(current_node, []):
            if neighbor not in visited:
                heapq.heappush(priority_queue, (heuristic_euclidean(neighbor, nodes, goals), neighbor, path + [neighbor]))

    return None, number_of_nodes, []