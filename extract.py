import fitz

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        full_text += text
    return full_text

text = extract_text_from_pdf("survey.pdf")
print(text[:500])
print(f"\n总字数：{len(text)}")