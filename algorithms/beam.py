import heapq
from algorithms.heuristic_euclidean import heuristic_euclidean

def beam(graph, nodes, start, goals, beam_width=2):
    priority_queue = []
    heapq.heappush(priority_queue, (heuristic_euclidean(start, nodes, goals), start, [start]))

    visited = set()
    number_of_nodes = 0

    while priority_queue:
        cost, current_node, path = heapq.heappop(priority_queue)

        if current_node in visited:
            continue
        visited.add(current_node)
        number_of_nodes += 1

        if current_node in goals:
            return current_node, number_of_nodes, path, cost

        for neighbor, _ in graph.get(current_node, []):
            if neighbor not in visited:
                heapq.heappush(priority_queue, (heuristic_euclidean(neighbor, nodes, goals), neighbor, path + [neighbor]))

        # Prune priority queue based on beam
        priority_queue = heapq.nsmallest(beam_width, priority_queue)

    return None, number_of_nodes, [], None