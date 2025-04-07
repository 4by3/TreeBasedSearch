import math

def heuristic_euclidean(node, nodes, goals):

    x1, y1 = nodes[node] 
    min_dist = float('inf') 

    for goal in goals:
        x2, y2 = nodes[goal]  # Coordinates of the goal node
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  # Euclidean formula
        min_dist = min(min_dist, dist)  # Keep the minimum distance

    return min_dist
