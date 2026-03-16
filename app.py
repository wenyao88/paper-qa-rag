from flask import Flask, request, jsonify, render_template_string
import requests
import faiss
import numpy as np
import pickle
from openai import OpenAI

app = Flask(__name__)

# 加载向量库
index = faiss.read_index("paper.index")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

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
    q_vector = np.array([get_embedding(question)]).astype('float32')
    distances, indices = index.search(q_vector, 5)
    relevant_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(relevant_chunks)
    prompt = f"""根据以下论文内容回答问题，用中文回答。

论文内容：
{context}

问题：{question}
"""
    response = client.chat.completions.create(
        model="internlm/internlm2_5-7b-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>论文问答助手</title>
    <meta charset="utf-8">
    <style>
        body { max-width: 800px; margin: 50px auto; font-family: Arial; padding: 20px; }
        input { width: 70%; padding: 10px; font-size: 16px; }
        button { padding: 10px 20px; font-size: 16px; background: #4CAF50; color: white; border: none; cursor: pointer; }
        #answer { margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; white-space: pre-wrap; }
        #loading { display: none; color: gray; margin-top: 10px; }
    </style>
</head>
<body>
    <h2>📄 论文问答助手</h2>
    <p>当前论文：Large Language Models Meet NLP: A Survey</p>
    <input type="text" id="question" placeholder="请输入你的问题..." />
    <button onclick="askQuestion()">提问</button>
    <div id="loading">思考中...</div>
    <div id="answer"></div>

    <script>
        async function askQuestion() {
            const q = document.getElementById('question').value;
            if (!q) return;
            document.getElementById('loading').style.display = 'block';
            document.getElementById('answer').innerText = '';
            const res = await fetch('/ask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: q})
            });
            const data = await res.json();
            document.getElementById('loading').style.display = 'none';
            document.getElementById('answer').innerText = data.answer;
        }
        document.getElementById('question').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') askQuestion();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/ask', methods=['POST'])
def ask_api():
    question = request.json.get('question')
    answer = ask(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True, port=5000)