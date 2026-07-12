def build_history_context(history: list[dict]) -> str:
    """
    Builds the history context string passed to the LLM for follow-up generation.
    Mirrors the master query slice exactly: it contains the input and output
    of every message whose input went into the master query.
    """
    anchor_idx = -1
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("agent_state") == "query":
            anchor_idx = i
            break
            
    if anchor_idx == -1:
        return ""
        
    context_slice = history[anchor_idx:]
    
    entries = []
    for i, turn in enumerate(context_slice, 1):
        # Only input and output are present in the history context.
        # Excludes: state, input_tokens, output_tokens, time_taken_s, and sources.
        user = turn.get("input", "")
        assistant = turn.get("output", "")
        entry = f"""<turn index="{i}">
  <user>{user}</user>
  <assistant>{assistant}</assistant>
</turn>"""
        entries.append(entry)
        
    return "\n".join(entries)
