import datetime

def log_interaction(user_text, reply, source, contexts=None):
    """
    Log chatbot interaction.
    If source == "rag", also log retrieved college names from metadata.
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Base log line
    log_line = f"[{ts}] USER: {user_text} | SOURCE: {source} | BOT: {reply}"

    # Add context details for RAG
    if source == "rag" and contexts:
        names = [c["metadata"].get("name", "Unknown") for c in contexts]
        log_line += f" | RAG colleges: {', '.join(names)}"

    # Write to file
    with open("chatbot.log", "a", encoding="utf-8") as f:
        f.write(log_line + "\n")

    # Also print to console
    print(log_line)
