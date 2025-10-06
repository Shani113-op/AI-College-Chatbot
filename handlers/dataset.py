from rapidfuzz import process, fuzz

def find_college_by_text(user_text, colleges, threshold=70):
    names = [c["name"] for c in colleges]
    match = process.extractOne(user_text, names, scorer=fuzz.WRatio)
    if match:
        name, score, idx = match
        if score >= threshold:
            return colleges[idx]
    return None

def college_info_from_query(user_text, college):
    text = user_text.lower()
    if "fee" in text:
        return f"{college['name']} fees: {college.get('fees', 'Not available')}"
    elif "placement" in text:
        return f"{college['name']} placement rate: {college.get('placementRate', 'Not available')}"
    elif "location" in text:
        return f"{college['name']} is located in {college.get('location', 'Unknown')}"
    elif "cutoff" in text:
        cutoffs = ", ".join([f"{k}: {v}" for k, v in college.get("cutOffs", {}).items()])
        return f"Cutoffs for {college['name']}: {cutoffs or 'Not available'}"
    elif "deadline" in text:
        return f"Admission deadline for {college['name']}: {college.get('admissionDeadline', 'Not available')}"
    elif "website" in text:
        return f"Website for {college['name']}: {college.get('website', 'Not available')}"
    return None
