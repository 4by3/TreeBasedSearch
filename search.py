from pathlib import Path
import sys
from parse_input import parse_input

# Import algorithms here
from algorithms.dfs import dfs
from algorithms.greedy_bfs import greedy_bfs
from algorithms.beam import beam
from algorithms.dijkstra import dijkstra
from algorithms.astar import astar

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <algorithm>\nAvailable algorithms: DFS, GreedyBFS, Dijkstra, Beam, Astar")
        sys.exit(1)

    filename = sys.argv[1]
    method = sys.argv[2].upper()
    
    if not Path(filename).exists():
        print(f"Error: File '{filename}' does not exist. Make sure to have 'tests/' before file name")
        sys.exit(1)

    graph, nodes, origin, goals = parse_input(filename)

    if method == "DFS":
        goal, number_of_nodes, path = dfs(graph, origin, goals)
    elif method == "DIJKSTRA":
        goal, number_of_nodes, path = dijkstra(graph, origin, goals)
    elif method == "GREEDYBFS":
        goal, number_of_nodes, path = greedy_bfs(graph, nodes, origin, goals)
    elif method == "BEAM":
        goal, number_of_nodes, path = beam(graph, nodes, origin, goals)
    elif method == "ASTAR":
        goal, number_of_nodes, path = astar(graph, nodes, origin, goals)
    else:
        print("Invalid algorithm")
        sys.exit(1)

    print(f"Filename: {filename} Method: {method}")
    if path:
        print(f"Goal: {goal} Total Nodes: {number_of_nodes}")
        print("Path: " + " → ".join(map(str, path)))
    else:
        print("No path found.")
