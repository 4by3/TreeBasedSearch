from pathlib import Path
import sys
from parse_search import parse_search

# Import algorithms here
from algorithms.dfs import dfs
from algorithms.bfs import bfs
from algorithms.greedy_bfs import greedy_bfs
from algorithms.beam import beam
from algorithms.dijkstra import dijkstra
from algorithms.astar import astar

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <algorithm>\nAvailable algorithms: DFS, BFS, GreedyBFS, Dijkstra, Astar, Beam")
        sys.exit(1)

    filename = sys.argv[1]
    method = sys.argv[2].upper()
    
    if not Path(filename).exists():
        print(f"Error: File '{filename}' does not exist. Make sure to have 'tests/' before file name")
        sys.exit(1)

    graph, nodes, origin, goals = parse_search(filename)

    # temp cost
    cost = 69

    if method == "DFS":
        goal, number_of_nodes, path, cost = dfs(graph, origin, goals)
    elif method == "BFS":
        goal, number_of_nodes, path, cost = bfs(graph, origin, goals)
    elif method == "DIJKSTRA":
        goal, number_of_nodes, path, cost = dijkstra(graph, origin, goals)
    elif method == "GREEDYBFS":
        goal, number_of_nodes, path, cost = greedy_bfs(graph, nodes, origin, goals, "E")
    elif method == "GREEDYBFSMANHATTAN":
        goal, number_of_nodes, path, cost = greedy_bfs(graph, nodes, origin, goals, "M")
    elif method == "BEAM":
        goal, number_of_nodes, path, cost = beam(graph, nodes, origin, goals, "E")
    elif method == "BEAMMANHATTAN":
        goal, number_of_nodes, path, cost = beam(graph, nodes, origin, goals, "M")
    elif method == "ASTAR":
        goal, number_of_nodes, path, cost = astar(graph, nodes, origin, goals, "E")
    elif method == "ASTARMANHATTAN":
        goal, number_of_nodes, path, cost = astar(graph, nodes, origin, goals, "M")
    else:
        print("Invalid algorithm")
        sys.exit(1)

    print(f"Filename: {filename} Method: {method}")
    if path:
        print(f"Goal: {goal} || Total Nodes: {number_of_nodes} || Cost: {cost}")
        print("Path: " + " → ".join(map(str, path)))
    else:
        print("No path found.")
