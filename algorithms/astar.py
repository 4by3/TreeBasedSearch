
import heapq
from algorithms.heuristic_euclidean import heuristic_euclidean

def astar(graph, nodes, start, goals, heuristic):

    if heuristic == "M":
        fHeuristic = heuristic_manhattan(start, nodes, goals)
    elif heuristic == "E":
        fHeuristic = heuristic_euclidean(start, nodes, goals)
        
    priority_queue = []    
    heapq.heappush(priority_queue, (fHeuristic, 0, start, [start]))
    
    number_of_nodes = 0
    visited = set()
    
    while priority_queue:
        fNode, cost, current_node, path = heapq.heappop(priority_queue)
        
        if current_node in visited:
            continue
        visited.add(current_node)
        number_of_nodes += 1
        
        if current_node in goals:
            return current_node, number_of_nodes, path, cost
        
        for neighbor, edge_cost in graph.get(current_node, []):
            if neighbor not in visited:
                if heuristic == "M":
                    edge_fNode = heuristic_manhattan(neighbor, nodes, goals) + cost + edge_cost
                elif heuristic == "E":
                    edge_fNode = heuristic_euclidean(neighbor, nodes, goals) + cost + edge_cost
                
                heapq.heappush(priority_queue, (edge_fNode, cost + edge_cost, neighbor, path + [neighbor]))
    
    return None, number_of_nodes, [], None
