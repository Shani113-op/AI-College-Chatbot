import requests, json

OLLAMA_URL = "http://localhost:11434/api/generate"

def ollama_chat(user_text, context=None, model="phi"):
    try:
        if context:
            prompt = f"""You are a helpful College Information Assistant.
Use this context to answer:

{context}

User: {user_text}
Answer clearly with facts from the context."""
        else:
            prompt = f"User: {user_text}\nAnswer as a helpful college assistant."

        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt},
            stream=True,
            timeout=60
        )

        full_reply = ""
        for line in response.iter_lines():
            if line:
                try:
                    part = json.loads(line.decode("utf-8"))
                    if "response" in part:
                        full_reply += part["response"]
                except:
                    pass
        return full_reply.strip() or None

    except Exception as e:
        print(f"⚠️ Ollama request failed: {e}")
        return None
