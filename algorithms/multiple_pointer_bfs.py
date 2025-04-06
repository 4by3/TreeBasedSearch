from collections import deque
import sys

def multiple_pointer_bfs(graph, start, goals):
    number_of_nodes = 0
    startDeque = deque([(start, [start])])
    startVisited = set()
    # Create an array containing every goal to expand
    goalsArr = []
    # Hashmap to find fastest goal
    goalsVisited = {}

    for goal in goals:
        goalsArr.append(deque([(goal, [goal])]))
    while startDeque:
        # Expand start
        node, path = startDeque.popleft()

        # If found goal
        if node in goalsVisited:
            goalPath = goalsVisited[node]

            # Making sure the path is valid in case goal path traversed a one way edge.
            if validatePath(graph, goalPath[::-1]):
                return goalPath[0], number_of_nodes, path + goalPath[::-1]
            print("Found one way path. Use MPBFS only on undirected graphs.\nInvalid Path: " + " → ".join(map(str, path + goalPath[::-1])))
            sys.exit(1)
        

        if node in startVisited:
            continue
        startVisited.add(node)
        number_of_nodes += 1

        # Use _ because we are not considering weight
        if node in graph:
            for currNode, _ in graph[node]:
                if currNode not in startVisited:
                    startDeque.append((currNode, path + []))


        # Expand all goals
        for goalDeque in goalsArr:
            if not goalDeque:
                continue
            goalNode, goalPath = goalDeque.popleft()
            if goalNode in goalsVisited:
                continue
            goalsVisited[goalNode] = goalPath
            number_of_nodes += 1
            if goalNode in graph:
                for currGoalNode, _ in graph[goalNode]:
                    if currGoalNode not in goalsVisited:
                        goalDeque.append((currGoalNode, goalPath + [currGoalNode]))
        

    return None, number_of_nodes, []  # No valid path found


# Looking if path is valid. If it's not, it means there was a one-way path on the goal path
def validatePath(graph, path):
    for i in range(len(path) - 1):
        valid = False
        for node, _ in graph[path[i]]:
            if path[i + 1] == node:
                valid = True
        if valid == False:
            return False
    return True