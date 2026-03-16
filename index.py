import requests
import faiss
import numpy as np
import pickle
from split import extract_text_from_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 第一步：切片
text = extract_text_from_pdf("survey.pdf")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)
print(f"共{len(chunks)}块，开始生成向量...")

# 第二步：批量获取向量（每次最多25个，避免超限）
def get_embeddings(texts):
    url = "https://api.siliconflow.cn/v1/embeddings"
    headers = {
        "Authorization": "Bearer sk-uovrzvonsmrntvtcxhsjdcujinhkiouuqgjifyjthmyxheng",
        "Content-Type": "application/json"
    }
    all_vectors = []
    for i in range(0, len(texts), 25):
        batch = texts[i:i+25]
        response = requests.post(url, headers=headers, json={
            "model": "BAAI/bge-m3",
            "input": batch
        })
        data = response.json()
        vectors = [item["embedding"] for item in data["data"]]
        all_vectors.extend(vectors)
        print(f"已处理{min(i+25, len(texts))}/{len(texts)}块")
    return all_vectors

vectors = get_embeddings(chunks)
vectors_np = np.array(vectors).astype('float32')

# 第三步：存入faiss
index = faiss.IndexFlatL2(len(vectors_np[0]))
index.add(vectors_np)
print(f"向量库建好了，共{index.ntotal}条")

# 第四步：保存到本地，下次不用重新算
faiss.write_index(index, "paper.index")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
print("保存完成：paper.index 和 chunks.pkl")