
import heapq
from algorithms.heuristic_euclidean import heuristic_euclidean

def astar(graph, nodes, start, goals):
    
    #need to do f(n) = g(n) + h(n)
    #f(n) = estimate distance from end node , g(n) = total cost so far to reach this node. , h(n) = heuristic score. \
    #Need to account for multiple goals: select the goal closest to the origin
    
    priority_queue = []    
    heapq.heappush(priority_queue, (heuristic_euclidean(start, nodes, goals), start, [start]))
    
    total_cost = 0
    number_of_nodes = 0
    visited = set()
    
    while priority_queue:
        node_cost, current_node, path = heapq.heappop(priority_queue)
        
        if current_node in visited:
            continue
        visited.add(current_node)
        number_of_nodes += 1
        
        if current_node in goals:
            return current_node, number_of_nodes, path, node_cost
        
        for neighbor, edge_cost in graph.get(current_node, []):
            if neighbor not in visited:
                heapq.heappush(priority_queue, (heuristic_euclidean(neighbor, nodes, goals) + edge_cost + total_cost, neighbor, path + [neighbor]))
    
    return None, number_of_nodes, []
                

        
    
    

    



