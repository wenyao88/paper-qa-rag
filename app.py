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
current_filename = None
paper_library = {}


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

    # 文件大小限制：50MB
    file.seek(0, 2)
    file_size_mb = file.tell() / (1024 * 1024)
    file.seek(0)
    if file_size_mb > 50:
        return jsonify({'error': f'文件过大（{file_size_mb:.1f}MB），请上传50MB以内的PDF'}), 400

    pdf_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    file.save(pdf_path)

    try:
        index, chunks, pages = build_index(pdf_path)
    except Exception as e:
        return jsonify({'error': f'处理失败：{str(e)}，请检查PDF是否可读'}), 500

    filename = file.filename
    paper_library[filename] = {'index': index, 'chunks': chunks, 'pages': pages}
    current_index, current_chunks, current_pages, current_filename = index, chunks, pages, filename

    return jsonify({
        'message': f'✅ 上传成功，共处理 {len(chunks)} 个文字块，可以开始提问了',
        'filename': filename,
        'paper_list': list(paper_library.keys())
    })

@app.route('/switch', methods=['POST'])
def switch():
    global current_index, current_chunks, current_pages, current_filename

    filename = request.json.get('filename')
    if not filename or filename not in paper_library:
        return jsonify({'error': '论文不存在'}), 404

    paper = paper_library[filename]
    current_index = paper['index']
    current_chunks = paper['chunks']
    current_pages = paper['pages']
    current_filename = filename

    return jsonify({'message': f'✅ 已切换到：{filename}'})


@app.route('/papers', methods=['GET'])
def get_papers():
    return jsonify({
        'paper_list': list(paper_library.keys()),
        'current': current_filename
    })

@app.route('/ask', methods=['POST'])
def ask():
    global current_index, current_chunks, current_pages

    if current_index is None:
        return jsonify({'answer': '⚠️ 请先上传PDF文件'}), 400

    # 问题为空判断
    data = request.get_json()
    if not data or not data.get('question', '').strip():
        return jsonify({'answer': '⚠️ 问题不能为空'}), 400
    question = data['question'].strip()

    # 问题长度限制
    if len(question) > 500:
        return jsonify({'answer': '⚠️ 问题太长，请控制在500字以内'}), 400

    try:
        # 检索
        q_vector = np.array(get_embedding([question])).astype('float32')
        distances, indices = current_index.search(q_vector, 5)
        relevant_chunks = [current_chunks[i] for i in indices[0]]
        relevant_pages = [current_pages[i] for i in indices[0]]

        # 相似度过滤：距离太大说明检索内容不相关
        min_distance = float(distances[0][0])
        if min_distance > 2.0:
            return jsonify({
                'answer': '⚠️ 未找到与问题相关的论文内容，请换个问法或确认论文是否包含相关内容',
                'sources': []
            })

        context = "\n\n".join(relevant_chunks)
        prompt = f"""你是一个论文问答助手。请严格根据下面提供的论文片段回答用户的问题，不要编造任何不在原文中的内容。如果提供的内容不足以回答问题，请直接说"根据论文内容无法回答这个问题"。用中文回答。

论文片段：
{context}

问题：{question}

请基于以上论文片段直接回答，不要说"根据您提供的内容"这类废话，直接给出答案。
"""
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            timeout=30  # 超时30秒
        )
        return jsonify({
            'answer': response.choices[0].message.content,
            'sources': list(set(relevant_pages))
        })

    except Exception as e:
        error_msg = str(e)
        if 'timeout' in error_msg.lower():
            return jsonify({'answer': '⚠️ 请求超时，请稍后重试'}), 503
        elif 'rate' in error_msg.lower():
            return jsonify({'answer': '⚠️ API调用频率超限，请稍等几秒再试'}), 429
        else:
            return jsonify({'answer': f'⚠️ 系统错误：{error_msg}'}), 500


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
        .progress-wrap { display:none; margin: 10px 0; }
        .progress-bar-bg { background:#e0e0e0; border-radius:8px; height:12px; overflow:hidden; }
        .progress-bar-fill { height:12px; border-radius:8px; background: linear-gradient(90deg,#4CAF50,#81C784);
            width:0%; transition: width 0.4s ease; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.7} }
        .stage-text { font-size:13px; color:#555; margin-top:6px; }
        .layout { display: flex; gap: 20px; }
        .sidebar { width: 220px; flex-shrink: 0; }
        .sidebar h4 { margin: 0 0 10px 0; font-size: 14px; color: #333; }
        .paper-item { padding: 8px 10px; margin-bottom: 6px; background: #fff; border: 1px solid #ddd;
            border-radius: 6px; cursor: pointer; font-size: 13px; word-break: break-all;
            transition: all 0.2s; }
        .paper-item:hover { border-color: #4CAF50; color: #4CAF50; }
        .paper-item.active { background: #e8f5e9; border-color: #4CAF50; color: #2e7d32; font-weight: bold; }
        .main-content { flex: 1; min-width: 0; }
    </style>
</head>
<body>
    <h2>📄 论文问答助手</h2>
    <div class="layout">
    <div class="sidebar">
        <h4>📚 已上传论文</h4>
        <div id="paperList"><p style="color:#aaa;font-size:13px">暂无论文</p></div>
    </div>
    <div class="main-content">

    <div class="upload-area">
        <p>上传你的PDF论文</p>
        <input type="file" id="pdfFile" accept=".pdf" />
        <br><br>
        <button onclick="uploadPDF()">上传并处理</button>
    </div>

    <div id="status"></div>
    <div class="progress-wrap" id="progressWrap">
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" id="progressFill"></div>
        </div>
        <div class="stage-text" id="stageText"></div>
    </div>

    <div id="qa-section">
        <input type="text" id="question" placeholder="请输入你的问题..." />
        <button onclick="askQuestion()">提问</button>
        <div id="answer"></div>
    </div>

    <script>
        async function uploadPDF() {
            const file = document.getElementById('pdfFile').files[0];
            if (!file) { alert('请先选择文件'); return; }

            // 显示进度条
            const progressWrap = document.getElementById('progressWrap');
            const progressFill = document.getElementById('progressFill');
            const stageText = document.getElementById('stageText');
            progressWrap.style.display = 'block';
            document.getElementById('status').innerText = '';
            document.getElementById('qa-section').style.display = 'none';

            // 模拟阶段进度（真实耗时在后端，前端给用户反馈）
            const stages = [
                { text: '📄 正在读取PDF文件...', width: '20%' },
                { text: '✂️ 正在切分文字块...', width: '40%' },
                { text: '🧠 正在生成向量（最慢的一步）...', width: '60%' },
                { text: '🗂️ 正在建立索引...', width: '85%' },
            ];
            let stageIdx = 0;
            const stageTimer = setInterval(() => {
                if (stageIdx < stages.length) {
                    stageText.innerText = stages[stageIdx].text;
                    progressFill.style.width = stages[stageIdx].width;
                    stageIdx++;
                }
            }, 2000);

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/upload', { method: 'POST', body: formData });
                const data = await res.json();
                clearInterval(stageTimer);

                if (data.error) {
                    progressFill.style.width = '100%';
                    progressFill.style.background = '#e53935';
                    stageText.innerText = '❌ ' + data.error;
                } else {
                    progressFill.style.width = '100%';
                    stageText.innerText = data.message;
                    document.getElementById('qa-section').style.display = 'block';
                    renderPaperList(data.paper_list, data.filename);
                }
            } catch(e) {
                clearInterval(stageTimer);
                stageText.innerText = '❌ 网络错误，请检查Flask是否在运行';
                progressFill.style.background = '#e53935';
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
        
        function renderPaperList(paperList, current) {
            const container = document.getElementById('paperList');
            if (!paperList || paperList.length === 0) {
                container.innerHTML = '<p style="color:#aaa;font-size:13px">暂无论文</p>';
                return;
            }
            container.innerHTML = paperList.map(name => `
                <div class="paper-item ${name === current ? 'active' : ''}"
                     onclick="switchPaper('${name}')"
                     title="${name}">
                    📄 ${name.length > 20 ? name.substring(0,20) + '...' : name}
                </div>
            `).join('');
        }

        async function switchPaper(filename) {
            const res = await fetch('/switch', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({filename})
            });
            const data = await res.json();
            if (data.message) {
                // 刷新列表高亮
                const res2 = await fetch('/papers');
                const d = await res2.json();
                renderPaperList(d.paper_list, d.current);
                document.getElementById('answer').innerText = '';
                document.getElementById('qa-section').style.display = 'block';
            }
        }
    </script>
    </div><!-- main-content -->
    </div><!-- layout -->
</body>
</html>
"""


@app.route('/')
def home():
    return render_template_string(HTML)


if __name__ == '__main__':
    app.run(debug=True, port=5000)