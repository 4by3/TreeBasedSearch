import heapq

# Time complexity is O(edges * log(nodes)) due to use of heap, which is longer compared to DFS and BFS
def dijkstra(graph, start, goals):
    # Set first value in priority queue to starting node
    priority_queue = [(0, start, [start])]
    visited = set()
    number_of_nodes = 0

    while priority_queue:
        cost, node, path = heapq.heappop(priority_queue)

        # Skip visited node
        if node in visited:
            continue
        visited.add(node)
        number_of_nodes += 1

        # Return path found
        if node in goals:
            return node, number_of_nodes, path, cost

        if node in graph:
            for neighbor, edge_cost in graph[node]:
                # Calculating total cost of this neighbour node by adding current node cost (total) with neighbour cost
                new_cost = cost + edge_cost
                if neighbor not in visited:
                    # Adding total cost in first so that heap sorts based on cost value
                    heapq.heappush(priority_queue, (new_cost, neighbor, path + [neighbor]))

    # Return no path found
    return None, number_of_nodes, [], None



# # Old
# import heapq

# filename = "./tests/PathFinder-test.txt"
# nodes, edges, origin, destinations = parse_file(filename)
# # Minheap begins on node 2 (1 according to arr)
# minHeap = [(0, 2 - 1)]
# visited = set()

# while minHeap:
#     weight, node = heapq.heappop(minHeap)
#     if node in visited:
#         print(f"Node {node} already visited")
#         continue
#     print(f"\nCurrently:\nAt node {node + 1} with current distance {weight}")
#     if any(node == destination for destination in destinations):
#         print(f"\nFinished\nGot to destination {node + 1} in {weight}.")
#         break
#     visited.add(node)
#     print("Adding node edges:")
#     for edge in edges[node]:
#         if edge[0] not in visited:
#             heapq.heappush(minHeap, (edge[1] + weight, edge[0]))
#             print(f"Pushed node {edge[0] + 1} with distance {edge[1] + weight}.")
#         else:
#             print(f"Node {edge[0] + 1} has already been visited.")
#     print("Priority Queue Size:")
#     for value in minHeap:
#         print(f"Node: {value[1]} Size: {value[0]}")