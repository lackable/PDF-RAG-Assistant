def build_master_query(history: list[dict], current_input: str) -> str:
    """
    Constructs the master query for retrieval on follow-up turns.
    Scans history in reverse to find the index of the last 'query' state message.
    Collects inputs from that anchor onwards, and appends the current_input.
    Joins all with " , ".
    If no 'query' state is found, falls back to returning just the current_input.
    """
    anchor_idx = -1
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("agent_state") == "query":
            anchor_idx = i
            break
            
    if anchor_idx == -1:
        # Edge case: no prior query state
        return current_input
        
    inputs = []
    for msg in history[anchor_idx:]:
        inp = msg.get("input", "").strip()
        if inp:
            inputs.append(inp)
        
    current = current_input.strip()
    if current:
        inputs.append(current)
    
    return " , ".join(inputs)
