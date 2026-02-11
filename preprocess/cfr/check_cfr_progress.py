# check_cfr_progress.py
"""Check progress of CFR JSON file"""
import json
from pathlib import Path

def check_progress():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    cfr_path = base_dir / "Data" / "Knowledge" / "cfr.json"
    
    if not cfr_path.exists():
        print("CFR JSON file not found")
        return
    
    try:
        with open(cfr_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        cfr = data.get("data", {}).get("code_of_federal_regulations", {})
        regs = cfr.get("regulations", cfr.get("titles", []))
        
        titles = set()
        for reg in regs:
            article = reg.get("article", "")
            if article:
                titles.add(article)
        
        print(f"Total regulations: {len(regs):,}")
        print(f"Titles found: {len(titles)}")
        if titles:
            sorted_titles = sorted(titles, key=lambda x: int(x.split()[-1]) if x.split()[-1].isdigit() else 999)
            print(f"Title numbers: {', '.join(sorted_titles[:10])}{'...' if len(sorted_titles) > 10 else ''}")
        
        # Count by title
        title_counts = {}
        for reg in regs:
            article = reg.get("article", "Unknown")
            title_counts[article] = title_counts.get(article, 0) + 1
        
        print("\nRegulations per title:")
        for title in sorted(title_counts.keys(), key=lambda x: int(x.split()[-1]) if x.split()[-1].isdigit() else 999):
            print(f"  {title}: {title_counts[title]:,} regulations")
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    check_progress()

