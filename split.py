import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        full_text += text

    if "References" in full_text:
        full_text = full_text[:full_text.rfind("References")]
    return full_text

text = extract_text_from_pdf("survey.pdf")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)

print(f"总共切了{len(chunks)}块")
print(f"\n第1块内容：\n{chunks[0]}")
print(f"\n第2块内容：\n{chunks[1]}")