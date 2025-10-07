import datetime
import re
from collections import deque

# ============================================================
# 📘 CHATBOT LOGGING & CONTEXT MODULE
# ============================================================

def log_interaction(user_text, reply, source, contexts=None, log_path="chatbot.log"):
    """
    Log chatbot interaction in a consistent, readable format.
    Automatically timestamps each entry and records source (dataset, ollama, rag, etc.).
    If RAG is used, it logs the retrieved document/college names for traceability.
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Base log line
    log_line = f"[{ts}] USER: {user_text} | SOURCE: {source} | BOT: {reply}"

    # Include retrieved RAG sources if available
    if source == "rag" and contexts:
        try:
            names = [c["metadata"].get("name", "Unknown") for c in contexts]
            if names:
                log_line += f" | RAG colleges: {', '.join(names)}"
        except Exception:
            log_line += " | RAG context: [ParseError]"

    # Write to persistent log file
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
    except Exception as e:
        print(f"[LOGGER ERROR] Could not write to log file: {e}")

    # Mirror log to console for live monitoring
    print(f"[LOGGED] {log_line}")


# ============================================================
# 🧠 CHAT CONTEXT MEMORY READER
# ============================================================

def read_chat_context(log_path="chatbot.log", max_entries=6):
    """
    Reads the last few user-bot exchanges from the chat log
    and formats them into a clean conversation-style text block.

    This context is passed into Ollama for continuity and reasoning.
    """
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return ""

    # Keep a rolling memory of the last N user-bot pairs
    user_bot_pairs = deque(maxlen=max_entries)
    user, bot = None, None

    for line in reversed(lines):
        user_match = re.search(r"USER:\s*(.*?)\s*\|", line)
        bot_match = re.search(r"BOT:\s*(.*)", line)
        if user_match:
            user = user_match.group(1)
        if bot_match:
            bot = bot_match.group(1)
        if user and bot:
            user_bot_pairs.appendleft((user.strip(), bot.strip()))
            user, bot = None, None

    # Build a readable conversation transcript
    if not user_bot_pairs:
        return ""

    conversation = ""
    for u, b in user_bot_pairs:
        conversation += f"User: {u}\nAssistant: {b}\n"

    return conversation.strip()


# ============================================================
# ✅ USAGE EXAMPLES
# ============================================================

if __name__ == "__main__":
    # Example logging
    log_interaction("fee of IIT Bombay", "IIT Bombay fees are ₹2,00,000/year.", "dataset")

    # Example reading
    context = read_chat_context(max_entries=5)
    print("\n--- Recent Chat Context ---")
    print(context)
