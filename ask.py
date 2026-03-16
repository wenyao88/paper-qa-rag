import requests
import faiss
import numpy as np
import pickle
from openai import OpenAI

# 加载向量库和文字块
index = faiss.read_index("paper.index")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# 硅基流动客户端
client = OpenAI(
    api_key="sk-uovrzvonsmrntvtcxhsjdcujinhkiouuqgjifyjthmyxheng",
    base_url="https://api.siliconflow.cn/v1"
)


def get_embedding(text):
    url = "https://api.siliconflow.cn/v1/embeddings"
    headers = {
        "Authorization": "Bearer sk-uovrzvonsmrntvtcxhsjdcujinhkiouuqgjifyjthmyxheng",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json={
        "model": "BAAI/bge-m3",
        "input": text
    })
    return response.json()["data"][0]["embedding"]


def ask(question):
    # 第一步：把问题转成向量
    q_vector = np.array([get_embedding(question)]).astype('float32')

    # 第二步：从向量库检索最相关的3块
    distances, indices = index.search(q_vector, 5)
    relevant_chunks = [chunks[i] for i in indices[0]]

    # 第三步：拼成prompt
    context = "\n\n".join(relevant_chunks)
    prompt = f"""根据以下论文内容回答问题，用中文回答。

论文内容：
{context}

问题：{question}
"""

    # 第四步：调大模型
    response = client.chat.completions.create(
        model="internlm/internlm2_5-7b-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# 开始问答
while True:
    question = input("\n请输入问题（输入exit退出）：")
    if question == "exit":
        break
    answer = ask(question)
    print(f"\n回答：{answer}")