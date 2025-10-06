# scripts/convert_js_to_json.py
import json
import re
import os

SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "staticColleges.js")
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "staticColleges.json")

def main():
    if not os.path.exists(SRC):
        print(f"⚠️ File {SRC} not found. Place staticColleges.js in project root.")
        return

    with open(SRC, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Remove JS export prefix
    content = re.sub(r"^export\s+const\s+\w+\s*=\s*", "", content)

    # Remove trailing semicolon
    if content.endswith(";"):
        content = content[:-1]

    # Remove trailing commas
    content = re.sub(r",(\s*[}\]])", r"\1", content)

    try:
        data = json.loads(content)
    except Exception as e:
        print("❌ Failed to parse JSON:", e)
        return

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Converted {SRC} → {OUT} with {len(data)} colleges")

if __name__ == "__main__":
    main()
