import os
import json

# ============================================================
# 💾 SINGLE GLOBAL MEMORY (memory.json)
# ============================================================

BASE_DIR = "logs"
os.makedirs(BASE_DIR, exist_ok=True)
MEMORY_FILE = os.path.join(BASE_DIR, "memory.json")


def load_memory() -> dict:
    """Load persistent memory from logs/memory.json."""
    if not os.path.exists(MEMORY_FILE):
        # Initialize empty memory
        return {
            "user_name": None,
            "college_name": None,
            "college_info": None
        }

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = {}

    # Ensure all expected keys exist
    data.setdefault("user_name", None)
    data.setdefault("college_name", None)
    data.setdefault("college_info", None)
    return data


def save_memory(memory: dict):
    """Save memory dictionary persistently to logs/memory.json."""
    os.makedirs(BASE_DIR, exist_ok=True)
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[MEMORY WARNING] Could not save memory: {e}")


def reset_memory():
    """Delete/reset memory.json."""
    try:
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)
            print("[MEMORY] memory.json has been reset.")
    except Exception as e:
        print(f"[MEMORY WARNING] Could not reset memory: {e}")


