def format_college_summary(metadata: dict) -> str:
    """
    Convert college metadata into a short, readable summary.
    Example: "MIT, located in Cambridge, offers Engineering and CS."
    """
    name = metadata.get("name", "Unknown College")
    location = metadata.get("location", "Unknown Location")
    courses = ", ".join(metadata.get("courses", []))
    return f"{name}, located in {location}. Courses offered: {courses}"
