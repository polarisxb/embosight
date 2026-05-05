import sys
from pypdf import PdfReader

reader = PdfReader("CRAIC2026.pdf")
print(f"PAGES: {len(reader.pages)}")
out = []
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    out.append(f"\n===== PAGE {i+1} =====\n{text}")
content = "\n".join(out)
with open("CRAIC2026.txt", "w", encoding="utf-8") as f:
    f.write(content)
print(f"WROTE {len(content)} chars")
