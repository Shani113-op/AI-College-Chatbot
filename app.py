# app.py — final production-ready (unified chatbot.log, long/descriptive expansion, full pipeline)
import os
import re
import json
import pickle
import datetime
from difflib import get_close_matches
from flask import Flask, render_template, request, jsonify

from handlers.ollama import ollama_chat
from handlers.nlp import classify_intent, get_default_response
from handlers.rag_handler import safe_retrieve_context
from utils.logger import log_interaction, read_chat_context
from utils.memory_helper import load_memory, save_memory



# ---- Config ----
MODEL_PATH = "models/classifier.pkl"
INTENTS_PATH = "intents/intents_examples.json"
COLLEGES_PATH = "data/staticColleges.json"

PRIMARY_MODEL = "phi"
FALLBACK_MODEL = "phi"
CONFIDENCE_THRESHOLD = 0.65

# ============================================================
# 🧠 AUTO-INITIALIZE GLOBAL MEMORY ON STARTUP
# ============================================================
try:
    memory = load_memory()
    # ensure base structure exists
    memory.setdefault("user_name", None)
    memory.setdefault("college_name", None)
    memory.setdefault("college_info", None)
    save_memory(memory)
    print("✅ memory.json initialized successfully.")
except Exception as e:
    print(f"⚠ Could not initialize memory.json: {e}")

# ---- Load classifier ----
with open(MODEL_PATH, "rb") as f:
    classifier, VOCAB = pickle.load(f)

# ---- Load intents ----
with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)["intents"]

# ---- Load static dataset ----
with open(COLLEGES_PATH, "r", encoding="utf-8") as f:
    COLLEGES = json.load(f)

app = Flask(__name__)

# -----------------------------------------------------------
# Unified logging for prompts & interactions (chatbot.log)
# -----------------------------------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
UNIFIED_LOG_PATH = os.path.join(LOG_DIR, "chatbot.log")

def log_ollama_prompt(prompt_text: str, model_name: str, source: str):
    """Log every prompt sent to Ollama into the unified chatbot.log file."""
    try:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        snippet = (prompt_text or "").replace("\n", " ").strip()
        if len(snippet) > 4000:
            snippet = snippet[:4000] + " ...[truncated]"
        with open(UNIFIED_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] [PROMPT-{source.upper()}] [{model_name}] {snippet}\n")
    except Exception as e:
        print(f"[LOGGING WARNING] Could not log Ollama prompt: {e}")

# -----------------------------------------------------------
# Globals & helpers
# -----------------------------------------------------------

# -----------------------------------------------------------
# 🧩 Load ABBREVIATIONS + optional expand_abbreviations function
# -----------------------------------------------------------
try:
    import importlib
    abbrev_module = importlib.import_module("abbreviations_phonetic")
    ABBREVIATIONS = getattr(abbrev_module, "ABBREVIATIONS", {})
    expand_abbreviations = getattr(abbrev_module, "expand_abbreviations", None)
    print(f"✅ Loaded {len(ABBREVIATIONS)} abbreviations from abbreviations_phonetic.py")
except Exception as e:
    print(f"⚠ Could not load ABBREVIATIONS: {e}")
    ABBREVIATIONS = {}
    expand_abbreviations = None



COURSE_ALIASES = {
    "btech": "B.Tech",
    "b.e": "B.E.",
    "be": "B.E.",
    "engineering": "Engineering",
    "computer science": "Computer Science",
    "mba": "MBA",
    "management": "Management",
    "bachelor of engineering": "B.E."
}

memory = {"college_name": None, "college_info": None}
last_greeting = {"text": ""}
greeting_keywords = ["hi", "hello", "hey", "good morning", "good evening"]
vague_queries = [
    "tell me more", "more info", "details", "explain", "continue",
    "information about", "describe", "overview", "more about", "what else"
]

# -----------------------------------------------------------
# General-question detector (so we don't attach college context)
# -----------------------------------------------------------
def is_general_question(text: str) -> bool:
    text = (text or "").lower()
    general_keywords = [
        "what is", "who is", "when is", "where is", "national", "president",
        "prime minister", "minister", "capital", "population", "python code",
        "program", "define", "explain", "history", "how to", "example", "code",
        "fibonacci", "flower", "animal", "language", "currency", "who", "what",
        "when", "how", "why"
    ]
    return any(k in text for k in general_keywords)

# -----------------------------------------------------------
# Preprocess query -> expand abbreviations and canonicalize
# -----------------------------------------------------------
def preprocess_query(user_text: str) -> str:
    text = (user_text or "").lower().strip()

    # ✅ Use expand_abbreviations() from abbreviations_phonetic.py if it exists
    if expand_abbreviations:
        text = expand_abbreviations(text)
    else:
        for abbr, full in ABBREVIATIONS.items():
            pattern = r"\b" + re.escape(abbr.lower()) + r"\b"
            text = re.sub(pattern, full.lower(), text)
    # extra mappings
    text = text.replace("iit bombay", "indian institute of technology bombay")
    text = text.replace("iitb", "indian institute of technology bombay")
    text = text.replace("vjti", "veermata jijabai technological institute")
    text = text.replace("du ", "delhi university ")
    text = text.replace("vit vellore", "vellore institute of technology")
    text = text.replace("vit ", "vellore institute of technology ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------------------------------------
# Normalize course alias
# -----------------------------------------------------------
def normalize_course(user_text: str):
    user_text_lower = (user_text or "").lower()
    for alias, normalized in COURSE_ALIASES.items():
        if alias in user_text_lower:
            return normalized
    return None

# -----------------------------------------------------------
# Pretty format for a single college
# -----------------------------------------------------------
def format_college_info(college_dict: dict) -> str:
    return (
        f"{college_dict.get('name','N/A')} is located in {college_dict.get('location','N/A')}, "
        f"ranked {college_dict.get('ranking','N/A')}, offers {college_dict.get('course','N/A')}. "
        f"It has a placement rate of {college_dict.get('placementRate','N/A')} "
        f"with top recruiters like {', '.join(college_dict.get('topRecruiters', [])[:3]) if college_dict.get('topRecruiters') else 'N/A'}. "
        f"Fees: {college_dict.get('fees','N/A')}. Entrance Exam: {college_dict.get('entranceExam','N/A')}. "
        f"Website: {college_dict.get('website','N/A')}."
    )

# -----------------------------------------------------------
# Cutoff helper
# -----------------------------------------------------------
def get_cutoff_response(college: dict, user_text: str) -> str:
    cutoffs = college.get("cutOffs", {})
    if not cutoffs:
        return f"Sorry, cutoff information for {college['name']} is not available."
    branch_match = [b for b in cutoffs.keys() if b.lower() in (user_text or "").lower()]
    if branch_match:
        branch = branch_match[0]
        return f"The cutoff for {branch} at {college['name']} is {cutoffs[branch]}."
    summary = ", ".join([f"{b}: {v}" for b, v in list(cutoffs.items())[:5]])
    return f"Here are some cutoff highlights for {college['name']}: {summary}."

# -----------------------------------------------------------
# Top colleges by city
# -----------------------------------------------------------
def get_top_colleges_by_city(city: str, dataset: list, top_n: int = 10, course: str = None) -> str:
    def safe_rank(value):
        try: return int(value)
        except (ValueError, TypeError): return 9999
    city_colleges = [c for c in dataset if city.lower() in c.get("location", "").lower()]
    if course:
        course_lower = course.lower()
        city_colleges = [c for c in city_colleges if course_lower in c.get("course", "").lower()]
    if not city_colleges:
        return f"Sorry, I couldn’t find colleges in {city.title()}" + (f" offering {course}" if course else "") + "."
    city_colleges.sort(key=lambda x: safe_rank(x.get('ranking')))
    top = city_colleges[:top_n]
    lines = []
    for i, c in enumerate(top, 1):
        cutoff_sample = next(iter(c.get("cutOffs", {}).items()), ("N/A", "N/A"))
        recruiters = ", ".join(c.get("topRecruiters", [])[:3]) or "N/A"
        lines.append(
            f"{i}. 🏫 {c['name']} (Rank: {c.get('ranking','N/A')})\n"
            f"   📍 Location: {c.get('location','N/A')}\n"
            f"   💸 Fees: {c.get('fees','N/A')}\n"
            f"   📈 Placement: {c.get('placementRate','N/A')}\n"
            f"   💼 Top Recruiters: {recruiters}\n"
            f"   🎯 Entrance Exam: {c.get('entranceExam','N/A')}\n"
            f"   🔢 Example Cutoff: {cutoff_sample[0]} - {cutoff_sample[1]}\n"
        )
    return f"🏙 Top {len(top)} Colleges in {city.title()}" + (f" offering {course}" if course else "") + ":\n\n" + "\n".join(lines)

# -----------------------------------------------------------
# Compare multiple colleges (structured dataset listing)
# -----------------------------------------------------------
def compare_multiple_colleges(query: str, dataset: list) -> str:
    query_proc = preprocess_query(query)
    matched_colleges = []
    parts = re.split(r"\s+vs\s+|\s+and\s+|,", query_proc)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = get_close_matches(part, [c["name"].lower() for c in dataset], n=1, cutoff=0.4)
        if match:
            college = next(c for c in dataset if c["name"].lower() == match[0])
            if college not in matched_colleges:
                matched_colleges.append(college)
    if not matched_colleges:
        return "Sorry, I couldn't identify the colleges to compare."
    field_map = {"fee": "fees", "tuition": "fees", "placement": "placementRate", "exam": "entranceExam", "entrance": "entranceExam", "cutoff": "cutOffs"}
    field = next((v for k, v in field_map.items() if k in query_proc), None)
    comparison_text = "📊 College Comparison:\n\n"
    for c in matched_colleges:
        comparison_text += f"🏫 {c.get('name','N/A')}\n"
        if field:
            if field == "cutOffs":
                comparison_text += f"- Cutoff sample: {', '.join(list(c.get('cutOffs', {}).keys())[:3]) or 'N/A'}\n"
            else:
                comparison_text += f"- {field.capitalize()}: {c.get(field,'N/A')}\n"
        else:
            comparison_text += (
                f"- 📍 Location: {c.get('location','N/A')}\n"
                f"- 💰 Fees: {c.get('fees','N/A')}\n"
                f"- 🎯 Placement Rate: {c.get('placementRate','N/A')}\n"
                f"- 💼 Top Recruiters: {', '.join(c.get('topRecruiters', [])) if c.get('topRecruiters') else 'N/A'}\n"
                f"- 🏆 Ranking: {c.get('ranking','N/A')}\n"
                f"- 🎯 Entrance Exam: {c.get('entranceExam','N/A')}\n"
            )
        comparison_text += "\n"
    return comparison_text

# -----------------------------------------------------------
# Generate Ollama-based side-by-side comparison explanation
# -----------------------------------------------------------
def generate_comparison_explanation(query: str, dataset: list) -> str:
    query_proc = preprocess_query(query)
    parts = re.split(r"\s+vs\s+|,|and|\s+or\s+", query_proc)
    matched = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = get_close_matches(part, [c["name"].lower() for c in dataset], n=1, cutoff=0.4)
        if match:
            college = next(c for c in dataset if c["name"].lower() == match[0])
            if college not in matched:
                matched.append(college)
    if len(matched) < 2:
        return None

    rows = []
    for c in matched:
        rows.append({
            "name": c.get("name", "N/A"),
            "location": c.get("location", "N/A"),
            "ranking": c.get("ranking", "N/A"),
            "fees": c.get("fees", "N/A"),
            "placement": c.get("placementRate", "N/A"),
            "recruiters": ", ".join(c.get("topRecruiters", [])[:5]) or "N/A"
        })

    # Build table block
    table_lines = []
    for key, label in [("location", "📍 Location"), ("ranking", "🏆 Ranking"), ("fees", "💰 Fees"), ("placement", "📈 Placement Rate"), ("recruiters", "💼 Top Recruiters")]:
        values = " | ".join([r[key] for r in rows])
        table_lines.append(f"{label}: {values}")
    table_block = "\n".join(table_lines)

    prompt = (
        f"You are an expert college counselor. The user asked: '{query}'.\n\n"
        f"Present a clean side-by-side comparison (table-like) using only these facts:\n\n"
        f"{table_block}\n\n"
        f"Then write a short 'Summary:' paragraph explaining which is better and why, based strictly on ranking, placement, fees, and recruiters. Do NOT invent facts."
    )

    log_ollama_prompt(prompt, PRIMARY_MODEL, "ollama-comparison")
    reply = ollama_chat(prompt, context=None, model=PRIMARY_MODEL)
    return reply

# -----------------------------------------------------------
# Main extractor for college info
# -----------------------------------------------------------
def extract_college_info(user_text: str, dataset: list) -> str:
    user_text_proc = preprocess_query(user_text)
    college_names = [c["name"] for c in dataset]

    # Special handling: user asked about IIT location/fees/general -> prefer IIT Bombay if present
    if re.search(r"\biit\b", user_text.lower()):
        iitb_match = next((c for c in dataset if "bombay" in c.get("name","").lower() and "indian institute of technology" in c.get("name","").lower()), None)
        if iitb_match:
            text = user_text.lower()
            if "location" in text or "where" in text:
                return f"{iitb_match['name']} is located in {iitb_match.get('location','N/A')}."
            if "fee" in text or "tuition" in text:
                return f"{iitb_match['name']} has fees: {iitb_match.get('fees','N/A')}."
            if "placement" in text:
                return f"{iitb_match['name']} has a placement rate of {iitb_match.get('placementRate','N/A')} with top recruiters: {', '.join(iitb_match.get('topRecruiters', []))}."
            if re.search(r"\bfee[s]?\b.*\biit\b", text):
                iit_colleges = [c for c in dataset if "indian institute of technology" in c.get("name","").lower()]
                if iit_colleges:
                    fees_info = "\n".join([f"🏫 {c['name']} — {c.get('fees','N/A')}" for c in iit_colleges])
                    return f"Here are the fee details for IITs:\n{fees_info}"

    # Compare/which/better queries -> produce premium Ollama comparison if possible
    if any(k in user_text_proc for k in ["compare", "vs", "and", "difference", "better", "which is better"]):
        ollama_comp = generate_comparison_explanation(user_text_proc, dataset)
        if ollama_comp:
            return ollama_comp
        structured = compare_multiple_colleges(user_text_proc, dataset)
        if structured:
            return structured

    # Top N query
    if "top" in user_text_proc and "college" in user_text_proc and "in" in user_text_proc:
        words = user_text_proc.split()
        try:
            idx = words.index("in") + 1
            city = words[idx].capitalize() if idx < len(words) else None
        except ValueError:
            city = None
        course = normalize_course(user_text_proc)
        top_n = next((int(w) for w in words if w.isdigit()), 10)
        if city:
            return get_top_colleges_by_city(city, dataset, top_n=top_n, course=course)

    # Single-college fuzzy match
    match = get_close_matches(user_text_proc, college_names, n=1, cutoff=0.4)
    if match:
        college = next(c for c in dataset if c["name"] == match[0])
        memory["college_name"] = college["name"]
        memory["college_info"] = college
        text = user_text_proc.lower()

        if "ranking" in text:
            return f"{college['name']} is ranked {college.get('ranking','N/A')}."
        if "location" in text or "where" in text:
            return f"{college['name']} is located in {college.get('location','N/A')}."
        if "course" in text or "program" in text:
            return f"{college['name']} offers {college.get('course','N/A')}."
        if "faculty" in text or "professor" in text:
            return f"{college['name']}'s faculty: {college.get('faculty','N/A')}."
        if "placement" in text or "recruiter" in text:
            return f"{college['name']} has a placement rate of {college.get('placementRate','N/A')} with top recruiters: {', '.join(college.get('topRecruiters', []))}."
        if "fee" in text or "tuition" in text:
            return f"{college['name']} has fees: {college.get('fees','N/A')}."
        if "website" in text:
            return f"{college['name']} website: {college.get('website','N/A')}."
        if "cutoff" in text or "cut off" in text:
            return get_cutoff_response(college, text)
        if "exam" in text or "entrance" in text:
            return f"The entrance exam for {college['name']} is {college.get('entranceExam','N/A')}."

        return format_college_info(college)

    return None

# -----------------------------------------------------------
# Greeting detection
# -----------------------------------------------------------
def is_greeting(user_text: str) -> bool:
    return any(word in (user_text or "").lower() for word in greeting_keywords)

# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_reply():
    from datetime import datetime
    import re
    from difflib import get_close_matches
    from collections import deque

    user_text = (request.json.get("message") or "").strip()
    reply, source = None, None

    try:
        text_lower = user_text.lower()
        memory = load_memory()  # ✅ Load from memory.json
        user_name = memory.get("user_name")

        # ============================================================
        # 🧩 Load recent chat context from chatbot.log
        # ============================================================
        def read_chat_context(log_path="chatbot.log", max_entries=6):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except FileNotFoundError:
                return ""

            user_bot_pairs = deque(maxlen=max_entries)
            user, bot = None, None
            for line in reversed(lines):
                user_match = re.search(r"USER:\s*(.*)\s*\|", line)
                bot_match = re.search(r"BOT:\s*(.*)", line)
                if user_match:
                    user = user_match.group(1)
                if bot_match:
                    bot = bot_match.group(1)
                if user and bot:
                    user_bot_pairs.appendleft((user.strip(), bot.strip()))
                    user, bot = None, None

            # Build conversation string
            conversation = ""
            for u, b in user_bot_pairs:
                conversation += f"User: {u}\nAssistant: {b}\n"
            return conversation.strip()

        chat_context = read_chat_context("chatbot.log", max_entries=6)

        # ============================================================
        # 🧠 STEP 0: Greeting Priority — handled first
        # ============================================================
        if is_greeting(text_lower):
            if user_name:
                reply = f"Hi again, {user_name} 👋 How are you today?"
            else:
                reply = "Hi there! 👋 I'm your career and college guide assistant. What's your name?"
            source = "greeting"

            log_interaction(user_text, reply, source)
            return jsonify({"reply": reply, "source": source})

        # ============================================================
        # 🧠 STEP 1: Introductions (persistent memory)
        # ============================================================
        if "my name is" in text_lower:
            match = re.search(r"my name is\s+([A-Za-z]+)", text_lower)
            if match:
                user_name = match.group(1).capitalize()
                memory["user_name"] = user_name
                save_memory(memory)  # ✅ persist globally
                reply = f"Nice to meet you, {user_name}! 😊 How can I help you today? Are you exploring courses or colleges?"
            else:
                reply = "Nice to meet you! 😊 Could you tell me your name again?"
            source = "personal-intro"

            log_interaction(user_text, reply, source)
            return jsonify({"reply": reply, "source": source})

        # ============================================================
        # 🧠 STEP 2: “What is my name?” — factual recall only
        # ============================================================
        elif any(q in text_lower for q in ["what is my name", "tell me my name", "do you know my name", "my name?"]):
            if user_name:
                reply = f"Your name is {user_name} 😊. It’s great to have you back!"
            else:
                reply = "I don’t think you’ve told me your name yet — what’s your name?"
            source = "personal-memory"

            log_interaction(user_text, reply, source)
            return jsonify({"reply": reply, "source": source})

        # ============================================================
        # 🧍 STEP 3: Human / reflective queries
        # ============================================================
        elif any(h in text_lower for h in [
            "what am i doing here", "why am i here", "who are you",
            "can you help", "i need your help", "please help me",
            "help", "assist me", "where am i"
        ]):
            convo_prompt = (
                f"Recent chat:\n{chat_context}\n\n"
                f"The user said: '{user_text}'.\n"
                f"You are a friendly assistant. Respond kindly and conversationally — "
                f"no college or factual data. Ask how you can assist, warmly."
            )
            log_ollama_prompt(convo_prompt, PRIMARY_MODEL, "ollama-human")
            reply = ollama_chat(convo_prompt, context=None, model=PRIMARY_MODEL)
            source = "ollama-human"

        # ============================================================
        # 🎓 STEP 4: Course or career guidance queries
        # ============================================================
        elif any(q in text_lower for q in [
            "which course", "what course", "suggest course", "choose course",
            "career advice", "recommend course", "guide me", "find college", "search college"
        ]):
            convo_prompt = (
                f"Recent chat:\n{chat_context}\n\n"
                f"User: '{user_text}'\n\n"
                f"You are a smart career counselor AI. Ask about their interests if unclear, "
                f"and guide them towards the right field (engineering, management, design, etc). "
                f"Once they are specific, you may suggest exploring dataset-based college options."
            )
            log_ollama_prompt(convo_prompt, PRIMARY_MODEL, "ollama-guidance")
            reply = ollama_chat(convo_prompt, context=None, model=PRIMARY_MODEL)
            source = "ollama-guidance"

        # ============================================================
        # 🧍 STEP 5: Personal / identity queries
        # ============================================================
        elif any(p in text_lower for p in [
            "who am i", "do you know me", "about me", "remember me",
            "call me", "know about me"
        ]):
            convo_prompt = (
                f"Recent chat:\n{chat_context}\n\n"
                f"The user said: '{user_text}'.\n\n"
                f"Respond warmly, as a human-like AI, and clarify that you don't store private memory. "
                f"Do not mention colleges or datasets. Stay natural and empathetic."
            )
            log_ollama_prompt(convo_prompt, PRIMARY_MODEL, "ollama-personal")
            reply = ollama_chat(convo_prompt, context=None, model=PRIMARY_MODEL)
            source = "ollama-personal"

        # ============================================================
        # 🏫 STEP 6: College info expansion
        # ============================================================
        elif any(v in text_lower for v in vague_queries):
            target_name = None
            possible_names = [c["name"].lower() for c in COLLEGES]
            match = get_close_matches(text_lower, possible_names, n=1, cutoff=0.4)
            if match:
                target_name = match[0].title()
                college_info = next(c for c in COLLEGES if c["name"].lower() == match[0])
                context = format_college_info(college_info)
                prompt = (
                    f"Chat context:\n{chat_context}\n\n"
                    f"Generate a descriptive, factual overview about {target_name}. "
                    f"Use verified dataset info only. Include academics, campus, and placements.\n\n{context}"
                )
                log_ollama_prompt(prompt, PRIMARY_MODEL, "ollama-dataset")
                reply = ollama_chat(prompt, context=None, model=PRIMARY_MODEL)
                source = "ollama-dataset"

        # ============================================================
        # 🧠 STEP 7: Dataset query (fees, cutoff, placements)
        # ============================================================
        if not reply:
            dataset_reply = extract_college_info(user_text, COLLEGES)
            if dataset_reply:
                reply = dataset_reply
                source = "dataset"

        # ============================================================
        # 🔍 STEP 8: RAG-based contextual retrieval
        # ============================================================
        if not reply:
            contexts = safe_retrieve_context(user_text)
            if contexts:
                rag_context = f"{chat_context}\n\n{'\n'.join(contexts)}"
                reply = ollama_chat(user_text, context=rag_context, model=PRIMARY_MODEL)
                source = "rag"

        # ============================================================
        # 🤖 STEP 9: General Ollama fallback
        # ============================================================
        if not reply:
            general_prompt = (
                f"Here is the recent chat history:\n{chat_context}\n\n"
                f"The user now said: '{user_text}'.\n\n"
                f"Respond clearly, concisely, and helpfully, maintaining tone continuity from chat history."
            )
            log_ollama_prompt(general_prompt, PRIMARY_MODEL, "ollama-general")
            reply = ollama_chat(general_prompt, context=None, model=PRIMARY_MODEL)
            source = "ollama-general"

        # ============================================================
        # 🛟 STEP 10: Final Fallback
        # ============================================================
        if not reply:
            reply = "🙏 Sorry, I couldn’t process that. Could you rephrase your question?"
            source = "fallback"

        # ============================================================
        # ✅ Log and Save Chat
        # ============================================================
        save_memory(memory)
        log_interaction(user_text, reply, source)
        return jsonify({"reply": reply, "source": source})

    except Exception as e:
        log_interaction(user_text, f"Error: {str(e)}", "error")
        return jsonify({
            "reply": "⚠ Something went wrong. Please try again.",
            "source": "error"
        }), 500


if __name__ == "__main__":
    # use_reloader=False helps avoid sporadic WinError 10038 on Windows debug reloads
    app.run(debug=True, use_reloader=False)
