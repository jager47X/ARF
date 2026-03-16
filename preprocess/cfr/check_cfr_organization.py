# check_cfr_organization.py
"""Check if CFR JSON is organized by title 1-50"""
import re
from pathlib import Path


def check_organization():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    cfr_path = base_dir / "Data" / "Knowledge" / "cfr.json"

    if not cfr_path.exists():
        print("CFR JSON file not found")
        return

    print(f"Checking organization of {cfr_path}...")

    # Read multiple parts of file to check order
    with open(cfr_path, 'r', encoding='utf-8') as f:
        # Read first 500KB, middle, and end
        content_start = f.read(500000)
        f.seek(f.tell())  # Continue from where we left off
        file_size = cfr_path.stat().st_size
        if file_size > 1000000:
            f.seek(file_size // 2)
            content_middle = f.read(500000)
        else:
            content_middle = ""
        if file_size > 500000:
            f.seek(max(0, file_size - 500000))
            content_end = f.read(500000)
        else:
            content_end = ""

    content = content_start + content_middle + content_end

    # Extract unique title transitions
    articles = re.findall(r'"article":\s*"([^"]+)"', content)

    print(f"Found {len(articles)} article fields in sample")

    if articles:
        # Find unique titles and their first occurrence
        seen_titles = []
        for article in articles:
            if article.startswith("Title ") and article not in seen_titles:
                seen_titles.append(article)

        print(f"\nUnique titles found in order (first {min(20, len(seen_titles))}):")
        for i, title in enumerate(seen_titles[:20], 1):
            print(f"  {i}. {title}")

        # Extract title numbers
        title_numbers = []
        for title in seen_titles:
            try:
                title_num = int(title.split()[-1])
                title_numbers.append(title_num)
            except Exception:
                pass

        if title_numbers:
            print(f"\nTitle numbers in order: {title_numbers[:30]}")

            # Check if starts with 1 and is sequential
            if len(title_numbers) >= 5:
                if title_numbers[0] == 1:
                    print("[OK] File starts with Title 1")

                    # Check if sequential
                    is_sequential = True
                    for i in range(1, min(10, len(title_numbers))):
                        if title_numbers[i] != title_numbers[i-1] + 1:
                            is_sequential = False
                            break

                    if is_sequential and len(title_numbers) >= 10:
                        print("[OK] Titles appear to be organized sequentially 1-50!")
                    elif title_numbers[:10] == list(range(1, 11)):
                        print("[OK] First 10 titles are in order 1-10!")
                    else:
                        print("[?] Titles may be organized, but need more verification")
                else:
                    print(f"[X] File does NOT start with Title 1 (starts with Title {title_numbers[0]})")
            else:
                print("Not enough unique titles to verify organization")
        else:
            print("No title numbers found")
    else:
        print("No articles found in file")

if __name__ == "__main__":
    check_organization()

