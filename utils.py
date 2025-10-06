# utils.py
import json, os
from difflib import get_close_matches

DATA_PATH = "data/staticColleges.json"

if not os.path.exists(DATA_PATH):
    raise SystemExit(f"❌ Missing {DATA_PATH}. Run convert_js_to_json.py first.")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    COLLEGES = json.load(f)

COLLEGE_NAMES = [c["name"] for c in COLLEGES]

def find_college_by_text(text, cutoff=0.6):
    if not text:
        return None
    text = text.lower()
    # direct match
    for c in COLLEGES:
        if c["name"].lower() in text:
            return c
    # fuzzy match
    matches = get_close_matches(text, COLLEGE_NAMES, n=1, cutoff=cutoff)
    if matches:
        for c in COLLEGES:
            if c["name"] == matches[0]:
                return c
    return None

def format_college_summary(college):
    top_recr = ", ".join(college.get("topRecruiters", [])[:5]) or "NA"
    return (f"{college['name']} (Rank {college.get('ranking','NA')}) — "
            f"Location: {college.get('location','NA')}, "
            f"Courses: {college.get('course','NA')}, "
            f"Fees: {college.get('fees','NA')}, "
            f"Placement: {college.get('placementRate','NA')}, "
            f"Top recruiters: {top_recr}")
