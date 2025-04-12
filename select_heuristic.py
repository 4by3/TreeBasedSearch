def SelectHeuristic():
    manhattan = None
    
    while manhattan != "true" or "false":
        manhattan = input("type: \n'true' to use Manhattan heuristic \n'false' to use Euclidean heuristic\n").strip().capitalize()
        
        if(manhattan == "True"):
            return "M"
        elif(manhattan == "False"):
            return "E"
