import os
import json
from datetime import datetime

# ============================================================
# 💾 UNIFIED GLOBAL MEMORY (multi-user, persistent, timestamped)
# ============================================================

BASE_DIR = "logs"
os.makedirs(BASE_DIR, exist_ok=True)
MEMORY_FILE = os.path.join(BASE_DIR, "memory.json")


# ------------------------------------------------------------
# Load and Save Utilities
# ------------------------------------------------------------
def load_memory() -> dict:
    """Load persistent multi-user memory from logs/memory.json."""
    if not os.path.exists(MEMORY_FILE):
        return {}

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = {}

    # Ensure backward compatibility: if it's a single-user dict
    if "user_name" in data:
        # Wrap it into a multi-user structure under "Guest"
        data = {"Guest": data}

    return data


def save_memory(data: dict):
    """Save memory dictionary persistently to logs/memory.json."""
    os.makedirs(BASE_DIR, exist_ok=True)
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[MEMORY WARNING] Could not save memory: {e}")


# ------------------------------------------------------------
# User-level memory operations
# ------------------------------------------------------------
def get_user_memory(username: str):
    """
    Get or initialize a specific user's memory.
    Creates a new entry if it doesn't exist.
    """
    memory = load_memory()
    username = username.capitalize()

    if username not in memory:
        memory[username] = {
            "user_name": username,
            "college_name": None,
            "college_info": None,
            "branch": None,
            "year": None,
            "favorite_subject": None,
            "last_seen": None,
            "conversation_history": []
        }
        save_memory(memory)

    return memory, memory[username]


def update_last_seen(user_data: dict):
    """Update user's last seen timestamp and return days since last visit."""
    now = datetime.now()
    last_seen = user_data.get("last_seen")
    user_data["last_seen"] = now.isoformat()

    if not last_seen:
        return None
    try:
        last_dt = datetime.fromisoformat(last_seen)
        return (now - last_dt).days
    except Exception:
        return None


# ------------------------------------------------------------
# Reset or Cleanup
# ------------------------------------------------------------
def reset_memory():
    """Delete/reset the entire memory.json file."""
    try:
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)
            print("[MEMORY] memory.json has been reset.")
    except Exception as e:
        print(f"[MEMORY WARNING] Could not reset memory: {e}")
