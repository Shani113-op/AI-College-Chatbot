import re
from difflib import get_close_matches

def get_college_recommendations(user_text, COLLEGES):
    """
    Understands user intent and gives factual, dataset-grounded answers.
    Supports multiple exams (JEE, MHTCET, GATE, CAT, etc.).
    """

    user_lower = user_text.lower()
    intent = None
    location = None
    matched_cities = []
    percentage = None
    course = None
    exam = None

    # ============================================================
    # 🔍 Detect Intent
    # ============================================================
    if any(p in user_lower for p in ["can i get", "am i eligible", "could i get", "admission in"]):
        intent = "eligibility"
    elif any(p in user_lower for p in ["where is", "location of", "address of"]):
        intent = "location"
    elif any(p in user_lower for p in ["show", "find", "colleges in", "suggest college", "recommend"]):
        intent = "recommendation"
    else:
        intent = "general"

    # ============================================================
    # 📊 Extract Percentage / Percentile
    # ============================================================
    perc_match = re.search(r"(\d{2,3})\s*(%|percent|percentile)", user_lower)
    if perc_match:
        try:
            percentage = int(perc_match.group(1))
        except ValueError:
            percentage = None

    # ============================================================
    # 🎓 Extract Course / Degree
    # ============================================================
    for field in ["btech", "engineering", "mba", "management", "mca", "bsc", "msc", "phd", "law"]:
        if field in user_lower:
            course = field.upper()
            break

    # ============================================================
    # 🧾 Extract Exam Name
    # ============================================================
    for ex in ["jee", "mhtcet", "cet", "gate", "cat", "neet", "upsc", "xat", "tancet", "bitsat"]:
        if ex in user_lower.replace("-", "").replace(" ", ""):
            exam = ex.upper()
            break

    # ============================================================
    # 🧠 Auto-default exam/course for IIT queries
    # ============================================================
    if "iit" in user_lower:
        if not exam:
            exam = "JEE"
        if not course:
            course = "BTECH"

    # ============================================================
    # 📍 Region / City Detection
    # ============================================================
    location_keywords = {
        "maharashtra": ["mumbai", "pune", "nagpur", "nashik", "aurangabad"],
        "vidarbha": ["nagpur", "amravati", "akola"],
        "western maharashtra": ["pune", "kolhapur", "sangli"],
        "marathwada": ["aurangabad", "latur", "nanded"],
        "delhi": ["new delhi", "delhi"],
        "gujarat": ["ahmedabad", "surat", "vadodara"],
        "karnataka": ["bangalore", "mysore"],
    }
    for region, cities in location_keywords.items():
        if region in user_lower:
            location = region
            matched_cities = cities
            break
        for city in cities:
            if city in user_lower:
                location = city
                matched_cities = [city]
                break
        if location:
            break

    # ============================================================
    # 🏫 Find Specific College
    # ============================================================
    college_names = [c["name"].lower() for c in COLLEGES]
    matched_college = get_close_matches(user_lower, college_names, n=1, cutoff=0.6)

    # ============================================================
    # 📍 INTENT: Location lookup
    # ============================================================
    if intent == "location" and matched_college:
        college_name = matched_college[0]
        college = next(c for c in COLLEGES if c["name"].lower() == college_name)
        return f"{college['name']} is located in {college.get('location','N/A')}.", "location"

    # ============================================================
    # 🎯 INTENT: Eligibility check
    # ============================================================
    if intent == "eligibility" and matched_college:
        college_name = matched_college[0]
        college = next(c for c in COLLEGES if c["name"].lower() == college_name)

        # Check correct exam cutoff
        cutoff_value = None
        cutoffs = college.get("cutOffs", {})

        if exam and exam in cutoffs:
            cutoff_value = float(str(cutoffs[exam]).replace("%", "").strip())
        elif "overall" in cutoffs:
            cutoff_value = float(str(cutoffs["overall"]).replace("%", "").strip())

        # ✅ Eligibility Logic
        if percentage and cutoff_value:
            if percentage >= cutoff_value:
                reply = (
                    f"✅ With your {percentage}% in {exam or 'exam'}, you have a good chance of admission "
                    f"to {college['name']}.\n"
                    f"It’s located in {college.get('location','N/A')} and offers {college.get('course','N/A')}.\n"
                    f"Placement: {college.get('placementRate','N/A')} | Fees: {college.get('fees','N/A')}."
                )
            else:
                # Suggest alternatives automatically
                alternatives = [
                    c for c in COLLEGES
                    if location and location.lower() in c.get("location","").lower()
                    and c["name"].lower() != college_name
                    and exam and exam.lower() in str(c.get("exam","")).lower()
                ][:3]
                alt_text = "\n".join(
                    [f"• {a['name']} ({a.get('location','N/A')})" for a in alternatives]
                ) or "No strong alternatives found nearby."
                reply = (
                    f"❌ With {percentage}% in {exam or 'exam'}, it's unlikely you’ll get admission in {college['name']} "
                    f"(typical cutoff: {cutoff_value}%).\n\nHere are some alternatives in {location or 'Maharashtra'}:\n{alt_text}"
                )
        else:
            reply = (
                f"{college['name']} admits via {college.get('exam','entrance exams')}.\n"
                f"Please mention your exam name and score (like 'I got 90 percentile in MHTCET')."
            )
        return reply, "eligibility"

    # ============================================================
    # 🎓 INTENT: General Recommendations
    # ============================================================
    if intent == "recommendation" or not matched_college:
        filtered = COLLEGES
        if matched_cities:
            filtered = [c for c in filtered if any(city.lower() in c.get("location","").lower() for city in matched_cities)]
        if course:
            filtered = [c for c in filtered if course.lower() in c.get("course","").lower()]
        if exam:
            filtered = [c for c in filtered if exam.lower() in str(c.get("exam","")).lower()]

        top_results = filtered[:5]
        if not top_results:
            return (
                f"I couldn’t find verified data for that query. Could you specify the exam or city again?",
                "no-match"
            )

        reply_lines = []
        for i, c in enumerate(top_results, 1):
            reply_lines.append(
                f"{i}. 🏫 {c['name']} — {c.get('location','N/A')} "
                f"(Exam: {c.get('exam','N/A')}, Course: {c.get('course','N/A')}, "
                f"Placement: {c.get('placementRate','N/A')}, Fees: {c.get('fees','N/A')})"
            )
        reply = (
            f"Here are some top verified colleges "
            f"{'in ' + location.title() if location else ''} "
            f"{'for ' + exam if exam else ''} "
            f"{'with course ' + course if course else ''}:\n\n"
            + "\n".join(reply_lines)
        )
        return reply, "recommendation"

    # ============================================================
    # 🧩 Default Fallback
    # ============================================================
    return (
        "I’m not sure what you meant. Try asking like:\n"
        "• 'Can I get admission in VJTI with 90 percentile in MHTCET?'\n"
        "• 'Show me BTech colleges in Pune for JEE'\n"
        "• 'Where is IIT Bombay located?'",
        "fallback"
    )
