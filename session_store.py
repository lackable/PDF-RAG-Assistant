import os
import json
import uuid
import datetime

SESSIONS_DIR = "sessions"
INDEX_FILE = os.path.join(SESSIONS_DIR, "index.json")

# Ensure sessions directory exists
os.makedirs(SESSIONS_DIR, exist_ok=True)
if not os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)

def _get_utc_now_str() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"

def _read_index() -> list[dict]:
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def _write_index(index_data: list[dict]):
    tmp_file = INDEX_FILE + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)
    os.replace(tmp_file, INDEX_FILE)

def create_session(name: str | None = None) -> dict:
    session_id = str(uuid.uuid4())
    now_str = _get_utc_now_str()
    session_name = name if name else "New Chat"

    session_info = {
        "id": session_id,
        "name": session_name,
        "created_at": now_str,
        "updated_at": now_str
    }

    # Add to index
    index_data = _read_index()
    index_data.append(session_info)
    _write_index(index_data)

    # Create session file
    session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    session_data = {
        "id": session_id,
        "name": session_name,
        "created_at": now_str,
        "messages": []
    }
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)

    return session_info

def list_sessions() -> list[dict]:
    index_data = _read_index()
    # Sort by updated_at descending (newest first)
    index_data.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return index_data

def get_session(session_id: str) -> dict | None:
    session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    try:
        with open(session_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None

def get_history(session_id: str) -> list[dict]:
    session = get_session(session_id)
    if session:
        return session.get("messages", [])
    return []

def append_message(
    session_id: str,
    agent_state: str,
    input_text: str,
    output_text: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    time_taken_s: float | None = None,
    sources: list[dict] | None = None
) -> None:
    session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    try:
        with open(session_file, "r", encoding="utf-8") as f:
            session_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return

    now_str = _get_utc_now_str()
    
    # Append message
    session_data.setdefault("messages", []).append({
        "agent_state": agent_state,
        "input": input_text,
        "output": output_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "time_taken_s": time_taken_s,
        "sources": sources if sources is not None else []
    })

    # Update session file atomically
    tmp_file = session_file + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)
    os.replace(tmp_file, session_file)

    # Update index timestamp
    index_data = _read_index()
    for item in index_data:
        if item["id"] == session_id:
            item["updated_at"] = now_str
            break
    _write_index(index_data)

def rename_session(session_id: str, new_name: str) -> None:
    # Update index
    index_data = _read_index()
    for item in index_data:
        if item["id"] == session_id:
            item["name"] = new_name
            break
    _write_index(index_data)

    # Update session file
    session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    try:
        with open(session_file, "r", encoding="utf-8") as f:
            session_data = json.load(f)
        session_data["name"] = new_name
        
        tmp_file = session_file + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)
        os.replace(tmp_file, session_file)
    except (json.JSONDecodeError, FileNotFoundError):
        pass

def delete_session(session_id: str) -> None:
    # Remove from index
    index_data = _read_index()
    index_data = [item for item in index_data if item["id"] != session_id]
    _write_index(index_data)

    # Delete session file
    session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(session_file):
        os.remove(session_file)

def auto_name_from_message(input_text: str) -> str:
    cleaned = input_text.strip()
    if len(cleaned) > 40:
        return cleaned[:37] + "..."
    return cleaned
