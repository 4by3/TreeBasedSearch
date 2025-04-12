import heapq
from algorithms.heuristic_euclidean import heuristic_euclidean
from algorithms.heuristic_manhattan import heuristic_manhattan

def greedy_bfs(graph, nodes, start, goals, heuristic):

    if heuristic == "M":
        fHeuristic = heuristic_manhattan(start, nodes, goals)
    elif heuristic == "E":
        fHeuristic = heuristic_euclidean(start, nodes, goals)
        
    priority_queue = []
    heapq.heappush(priority_queue, (fHeuristic, 0, start, [start]))

    visited = set()
    number_of_nodes = 0

    while priority_queue:
        fNode, cost, current_node, path = heapq.heappop(priority_queue)

        if current_node in visited:
            continue
        visited.add(current_node)
        number_of_nodes += 1

        if current_node in goals:
            return current_node, number_of_nodes, path, cost

        for neighbor, node_cost in graph.get(current_node, []):
            if neighbor not in visited:

                if heuristic == "M":
                    cHeuristic = heuristic_manhattan(neighbor, nodes, goals)
                elif heuristic == "E":
                    cHeuristic = heuristic_euclidean(neighbor, nodes, goals)
                    
                heapq.heappush(priority_queue, (cHeuristic, cost + node_cost, neighbor, path + [neighbor]))

    return None, number_of_nodes, [], None
