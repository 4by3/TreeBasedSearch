def parse_search(filename):
    graph = {}
    nodes = {}
    origin = None
    goals = set()

    with open(filename, 'r') as file:
        section = None
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):  
                continue

            if line.startswith("Nodes:"):
                section = "nodes"
                continue
            elif line.startswith("Edges:"):
                section = "edges"
                continue
            elif line.startswith("Origin:"):
                section = "origin"
                continue
            elif line.startswith("Destinations:"):
                section = "goals"
                continue

            if section == "nodes":
                parts = line.split(":")
                node_id = int(parts[0].strip())
                coords = tuple(map(int, parts[1].strip(" ()").split(",")))
                nodes[node_id] = coords

            elif section == "edges":
                parts = line.split(":")
                edge_nodes = tuple(map(int, parts[0].strip(" ()").split(",")))
                cost = int(parts[1].strip())
                if edge_nodes[0] not in graph:
                    graph[edge_nodes[0]] = []
                graph[edge_nodes[0]].append((edge_nodes[1], cost))

            elif section == "origin":
                origin = int(line.strip())

            elif section == "goals":
                goals = set(map(int, line.split(";")))

    return graph, nodes, origin, goals