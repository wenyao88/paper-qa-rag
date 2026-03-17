from flask import Flask, request, jsonify, render_template_string
import requests
import faiss
import numpy as np
import pickle
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件
API_KEY = os.getenv("SILICONFLOW_API_KEY")  # 从环境变量读取

app = Flask(__name__)

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.siliconflow.cn/v1"
)

# 全局变量，存当前加载的索引和文字块
current_index = None
current_chunks = None
current_pages = None


def get_embedding(texts):
    url = "https://api.siliconflow.cn/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    all_vectors = []
    for i in range(0, len(texts), 25):
        batch = texts[i:i + 25]
        response = requests.post(url, headers=headers, json={
            "model": "BAAI/bge-m3",
            "input": batch
        })
        data = response.json()
        vectors = [item["embedding"] for item in data["data"]]
        all_vectors.extend(vectors)
    return all_vectors


def build_index(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    page_map = []  # 记录每块文字来自哪页

    for page_num, page in enumerate(doc):
        text = page.get_text()
        start = len(full_text)
        full_text += text
        page_map.append((start, len(full_text), page_num + 1))

    if "References" in full_text:
        full_text = full_text[:full_text.rfind("References")]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(full_text)

    # 给每块找到对应页码
    chunk_pages = []
    search_pos = 0
    for chunk in chunks:
        chunk_start = full_text.find(chunk, search_pos)
        if chunk_start == -1:
            chunk_start = full_text.find(chunk)  # 回退：从头搜
        if chunk_start == -1:
            chunk_pages.append(1)  # 实在找不到，默认第1页
            continue
        page_num = 1
        for start, end, pnum in page_map:
            if start <= chunk_start < end:
                page_num = pnum
                break
        chunk_pages.append(page_num)
        search_pos = chunk_start + 1

    vectors = get_embedding(chunks)
    vectors_np = np.array(vectors).astype('float32')
    index = faiss.IndexFlatL2(len(vectors_np[0]))
    index.add(vectors_np)

    return index, chunks, chunk_pages


@app.route('/upload', methods=['POST'])
def upload():
    global current_index, current_chunks, current_pages

    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400

    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': '只支持PDF文件'}), 400

    # 保存上传的文件
    pdf_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    file.save(pdf_path)

    # 建索引
    current_index, current_chunks, current_pages = build_index(pdf_path)

    return jsonify({'message': f'上传成功，共处理{len(current_chunks)}个文字块，可以开始提问了'})


@app.route('/ask', methods=['POST'])
def ask():
    global current_index, current_chunks, current_pages

    if current_index is None:
        return jsonify({'answer': '请先上传PDF文件'}), 400

    question = request.json.get('question')

    # 检索
    q_vector = np.array(get_embedding([question])).astype('float32')
    distances, indices = current_index.search(q_vector, 5)
    relevant_chunks = [current_chunks[i] for i in indices[0]]
    relevant_pages = [current_pages[i] for i in indices[0]]
    context = "\n\n".join(relevant_chunks)

    prompt = f"""你是一个论文问答助手。请严格根据下面提供的论文片段回答用户的问题，不要编造任何不在原文中的内容。如果提供的内容不足以回答问题，请直接说"根据论文内容无法回答这个问题"。用中文回答。

论文片段：
{context}

问题：{question}

请基于以上论文片段直接回答，不要说"根据您提供的内容"这类废话，直接给出答案。
"""
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[{"role": "user", "content": prompt}]
    )
    return jsonify({
        'answer': response.choices[0].message.content,
        'sources': list(set(relevant_pages))  # 去重后的页码列表
    })


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>论文问答助手</title>
    <meta charset="utf-8">
    <style>
        body { max-width: 800px; margin: 50px auto; font-family: Arial; padding: 20px; }
        .upload-area { border: 2px dashed #ccc; padding: 30px; text-align: center; margin-bottom: 20px; border-radius: 8px; }
        input[type="text"] { width: 70%; padding: 10px; font-size: 16px; }
        button { padding: 10px 20px; font-size: 16px; background: #4CAF50; color: white; border: none; cursor: pointer; border-radius: 4px; }
        #status { margin: 10px 0; color: gray; }
        #answer { margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; white-space: pre-wrap; }
        #qa-section { display: none; }
    </style>
</head>
<body>
    <h2>📄 论文问答助手</h2>

    <div class="upload-area">
        <p>上传你的PDF论文</p>
        <input type="file" id="pdfFile" accept=".pdf" />
        <br><br>
        <button onclick="uploadPDF()">上传并处理</button>
    </div>

    <div id="status"></div>

    <div id="qa-section">
        <input type="text" id="question" placeholder="请输入你的问题..." />
        <button onclick="askQuestion()">提问</button>
        <div id="answer"></div>
    </div>

    <script>
        async function uploadPDF() {
            const file = document.getElementById('pdfFile').files[0];
            if (!file) { alert('请先选择文件'); return; }

            document.getElementById('status').innerText = '处理中，请稍候（大文件可能需要1-2分钟）...';

            const formData = new FormData();
            formData.append('file', file);

            const res = await fetch('/upload', { method: 'POST', body: formData });
            const data = await res.json();

            document.getElementById('status').innerText = data.message || data.error;
            if (data.message) {
                document.getElementById('qa-section').style.display = 'block';
            }
        }

        async function askQuestion() {
            const q = document.getElementById('question').value;
            if (!q) return;
            document.getElementById('answer').innerText = '思考中...';
            const res = await fetch('/ask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: q})
            });
            const data = await res.json();
            document.getElementById('answer').innerText = data.answer;
            if (data.sources) {
                document.getElementById('answer').innerText += 
                    '\\n\\n📄 参考页码：第' + data.sources.join('、') + '页';
                    }
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


if __name__ == '__main__':
    app.run(debug=True, port=5000)