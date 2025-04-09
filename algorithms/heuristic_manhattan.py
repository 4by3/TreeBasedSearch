import math

def heuristic_manhattan(node, nodes, goals):

    x1, y1 = nodes[node] 
    min_dist = float('inf') 

    for goal in goals:
        x2, y2 = nodes[goal]  # Coordinates of the goal node
        dist =  abs(x2 - x1) + abs(y2 - y1) 
        min_dist = min(min_dist, dist)  # Keep the minimum distance

    return min_dist
