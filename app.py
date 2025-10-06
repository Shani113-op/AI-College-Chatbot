# app.py — final production-ready (dataset + Ollama hybrid, memory, compare, factual-mode)
import os
import re
import json
import pickle
from difflib import get_close_matches
from flask import Flask, render_template, request, jsonify

from handlers.ollama import ollama_chat
from handlers.nlp import classify_intent, get_default_response
from handlers.rag_handler import safe_retrieve_context
from utils.logger import log_interaction

# ---- Config ----
MODEL_PATH = "models/classifier.pkl"
INTENTS_PATH = "intents/intents_examples.json"
COLLEGES_PATH = "data/staticColleges.json"

PRIMARY_MODEL = "phi"
FALLBACK_MODEL = "phi"
CONFIDENCE_THRESHOLD = 0.65

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
# Globals & helpers
# -----------------------------------------------------------
ABBREVIATIONS = {
    "iitb": "Indian Institute of Technology Bombay",
    "iit bombay": "Indian Institute of Technology Bombay",
    "iit": "Indian Institute of Technology",
    "du": "Delhi University",
    "vjti": "Veermata Jijabai Technological Institute",
    "bits": "Birla Institute of Technology and Science",
    "vit": "Vellore Institute of Technology",
    "nit": "National Institute of Technology"
}

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
        "fibonacci", "flower", "animal", "language", "currency"
    ]
    return any(k in text for k in general_keywords)

# -----------------------------------------------------------
# Preprocess query -> expand abbreviations and canonicalize
# -----------------------------------------------------------
def preprocess_query(user_text: str) -> str:
    text = (user_text or "").lower().strip()

    # Expand common abbreviations/aliases first (word boundaries)
    for abbr, full in ABBREVIATIONS.items():
        pattern = r"\b" + re.escape(abbr.lower()) + r"\b"
        text = re.sub(pattern, full.lower(), text)

    # Extra replacements for common forms
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
        # Look for explicit IIT Bombay / IIT entries in the dataset
        iitb_match = next((c for c in dataset if "bombay" in c.get("name","").lower() and "indian institute of technology" in c.get("name","").lower()), None)
        if iitb_match:
            # Interpretable queries
            text = user_text.lower()
            if "location" in text or "where" in text:
                return f"{iitb_match['name']} is located in {iitb_match.get('location','N/A')}."
            if "fee" in text or "tuition" in text:
                return f"{iitb_match['name']} has fees: {iitb_match.get('fees','N/A')}."
            if "placement" in text:
                return f"{iitb_match['name']} has a placement rate of {iitb_match.get('placementRate','N/A')} with top recruiters: {', '.join(iitb_match.get('topRecruiters', []))}."
            # fallback to listing IIT fees if generic
            if re.search(r"\bfee[s]?\b.*\biit\b", text):
                # group IIT fee listing (all IITs in dataset)
                iit_colleges = [c for c in dataset if "indian institute of technology" in c.get("name","").lower()]
                if iit_colleges:
                    fees_info = "\n".join([f"🏫 {c['name']} — {c.get('fees','N/A')}" for c in iit_colleges])
                    return f"Here are the fee details for IITs:\n{fees_info}"

    # Compare/which/better queries -> produce premium Ollama comparison if possible
    if any(k in user_text_proc for k in ["compare", "vs", "and", "difference", "better", "which is better"]):
        # Try generating premium explanation with Ollama
        ollama_comp = generate_comparison_explanation(user_text_proc, dataset)
        if ollama_comp:
            return ollama_comp
        # fallback to structured dataset compare
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
    user_text = (request.json.get("message") or "").strip()
    reply, source = None, None

    try:
        # Step 0: Vague follow-up first (e.g., "tell me more")
        if any(v in user_text.lower() for v in vague_queries) and memory["college_info"]:
            context = format_college_info(memory["college_info"])
            prompt = (
                f"Based on the following official dataset information, generate a friendly, detailed, and accurate explanation about {memory['college_name']}.\n\n"
                f"{context}\n\n"
                f"Use only the information provided. Keep it conversational and helpful."
            )
            log_ollama_prompt(prompt, PRIMARY_MODEL, "ollama-dataset-memory")
            reply = ollama_chat(prompt, context=None, model=PRIMARY_MODEL)
            source = "ollama-dataset-memory"

        # Step 1: Greeting
        if not reply and is_greeting(user_text):
            if last_greeting["text"].lower() != user_text.lower():
                reply, source = "Hi there! 👋 How can I help you today?", "greeting"
                last_greeting["text"] = user_text
            else:
                reply, source = "You already said hi! 😄 How else can I assist you?", "greeting"

        # ---- Step 1.5: Context Expansion for lifestyle/campus queries ----
        if not reply and any(k in user_text.lower() for k in ["campus", "life", "infrastructure", "hostel", "environment", "culture"]):
    college_name = memory.get("college_name")
    if college_name and memory.get("college_info"):
        base_context = format_college_info(memory["college_info"])
        expansion_prompt = (
            f"You are a knowledgeable assistant specializing in Indian colleges.\n"
            f"Based on the verified data below, write a natural, friendly, and detailed paragraph describing "
            f"the *campus life, student culture, and facilities* of {college_name}. "
            f"Use the factual data only as a base (location, ranking, placement, courses, etc.) — "
            f"but you may add realistic, generic details (like hostels, fests, labs, environment) "
            f"that fit the tone of a real student review. Avoid repeating the same data lines verbatim.\n\n"
            f"--- Verified Data ---\n{base_context}\n\n"
            f"--- Output Example ---\n"
            f"VJTI offers an energetic campus experience with strong academic support, annual tech festivals, "
            f"modern labs, and active student clubs that foster innovation and creativity. "
            f"Now write the equivalent for {college_name}."
        )
        log_ollama_prompt(expansion_prompt, PRIMARY_MODEL, "ollama-context-expansion")
        reply = ollama_chat(expansion_prompt, context=None, model=PRIMARY_MODEL)
        source = "ollama-context-expansion"

        # Step 2: Dataset queries (direct match / compare / top / cutoff)
        if not reply:
            dataset_reply = extract_college_info(user_text, COLLEGES)
            if dataset_reply:
                reply = dataset_reply
                source = "dataset"

        # Step 3: RAG retrieval
        if not reply:
            contexts = safe_retrieve_context(user_text)
            if contexts:
                reply = ollama_chat(user_text, "\n".join(contexts), model=PRIMARY_MODEL)
                source = "rag"

        # Step 4: Ollama fallback (general questions)
        if not reply:
            # Attach context if relevant
            if memory["college_info"] and not is_general_question(user_text):
                context = format_college_info(memory["college_info"])
            else:
                context = None

            if is_general_question(user_text):
                # Factual mode to prevent dataset hallucination
                general_prompt = (
                    f"You are a concise and factual assistant. Answer the following question directly and succinctly. "
                    f"Do not reference unrelated dataset items or invent facts.\n\n"
                    f"Question: {user_text}"
                )
                log_ollama_prompt(general_prompt, PRIMARY_MODEL, "ollama-general")
                reply = ollama_chat(general_prompt, context=None, model=PRIMARY_MODEL)
            else:
                reply = ollama_chat(user_text, context=context, model=PRIMARY_MODEL) \
                      or ollama_chat(user_text, None, model=FALLBACK_MODEL)

            if reply:
                source = "ollama"

        # Step 5: Intent/NLTK fallback
        if not reply:
            intent, conf = classify_intent(user_text, classifier, VOCAB)
            if conf >= CONFIDENCE_THRESHOLD:
                reply = get_default_response(intent, intents)
                source = "nltk"

        # Step 6: Catch-all
        if not reply:
            reply, source = "🙏 Sorry, I couldn’t process that. Try rephrasing your question.", "fallback"

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
