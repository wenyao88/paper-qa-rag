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
chat_history = []

def hyde_query(question):
    """
    HyDE：Hypothetical Document Embeddings
    核心思想：用大模型生成假设答案，用假设答案的向量去检索，
    而不是用问题本身的向量。假设答案和论文原文语义空间更接近。
    """
    prompt = f"""你是一个学术论文专家。请根据下面的问题，生成一段简短的英文学术文本，
就像这个问题的答案直接出现在论文中一样。只生成英文文本，不要解释，不要说"假设"，直接写内容，2-4句话即可。

问题：{question}"""
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            timeout=15
        )
        hypothetical_answer = response.choices[0].message.content.strip()
        return hypothetical_answer
    except:
        return question  # 生成失败则退回原始问题

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

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
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

@app.route('/clear_history', methods=['POST'])
def clear_history():
    global chat_history
    chat_history = []
    return jsonify({'message': '对话历史已清空'})

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
    chat_history.clear()

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
        # 检索：多取几个候选，再过滤
        hyde_doc = hyde_query(question)  # 生成假设答案
        q_vector = np.array(get_embedding([hyde_doc])).astype('float32')
        distances, indices = current_index.search(q_vector, 15)

        # 最相关的一个距离太大，说明整篇论文都没有相关内容
        min_distance = float(distances[0][0])
        if min_distance > 2.0:
            return jsonify({
                'answer': '⚠️ 未找到与问题相关的论文内容，请换个问法或确认论文是否包含相关内容',
                'sources': []
            })

        # 重排序：只保留距离小于阈值的chunk，最多取5个
        DISTANCE_THRESHOLD = 1.5
        filtered = [
            (distances[0][i], indices[0][i])
            for i in range(len(indices[0]))
            if distances[0][i] < DISTANCE_THRESHOLD
        ]
        # 如果过滤后不足3个，放宽阈值取前3个保底
        if len(filtered) < 3:
            filtered = [(distances[0][i], indices[0][i]) for i in range(5)]

        # 按距离从小到大排序（最相关的排前面）
        filtered.sort(key=lambda x: x[0])
        top = filtered[:7]

        relevant_chunks = [current_chunks[idx] for _, idx in top]
        relevant_pages = [current_pages[idx] for _, idx in top]

        context = "\n\n".join(relevant_chunks)

        # 构建系统提示
        system_prompt = f"""你是一个严谨的学术论文问答助手。

        【强制规则】
        1. 只能使用下方"论文片段"中明确出现的信息回答问题
        2. 数字、指标、方法名必须与原文完全一致，不得修改或推断
        3. 如果片段中没有足够信息，直接回答"论文提供的片段中未包含该信息"
        4. 禁止结合任何外部知识进行补充或推断
        5. 回答时注明信息来自哪个具体论点或实验

        【论文片段】
        {context}

        【回答格式】
        - 直接给出答案，不要说"根据论文"等废话
        - 如涉及数据，用"论文指出：XXX"的形式明确标注是原文内容
        - 结合对话历史理解追问意图"""

        # 组装消息：系统提示 + 历史对话 + 当前问题
        messages = [{"role": "system", "content": system_prompt}]
        # 只保留最近3轮历史，避免token超限
        for h in chat_history[-3:]:
            messages.append({"role": "user", "content": h['question']})
            messages.append({"role": "assistant", "content": h['answer']})
        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=messages,
            timeout=30
        )
        answer = response.choices[0].message.content

        # 存入历史
        chat_history.append({'question': question, 'answer': answer})

        return jsonify({
            'answer': answer,
            'sources': list(set(relevant_pages)),
            'history_count': len(chat_history)
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
<html lang="zh">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>论文问答助手</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400&display=swap" rel="stylesheet">
    <style>
        :root {
            --sidebar-bg:    #f4f2ee;
            --sidebar-border:#e5e2db;
            --main-bg:       #faf9f7;
            --white:         #ffffff;
            --border:        #e8e5df;
            --text-primary:  #1c1917;
            --text-secondary:#6b6560;
            --text-muted:    #b0aba4;
            --accent:        #c17f3e;
            --accent-light:  #fef3e6;
            --accent-dark:   #9d6530;
            --accent-border: #f0d5b0;
            --user-bg:       #eef2ff;
            --user-border:   #c7d2fe;
            --shadow-sm:     0 1px 3px rgba(0,0,0,0.05);
            --shadow-md:     0 4px 16px rgba(0,0,0,0.07);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'DM Sans', sans-serif;
            font-weight: 400;
            background: var(--main-bg);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            overflow: hidden;
        }

        /* ══ 侧边栏 ══ */
        .sidebar {
            width: 252px;
            min-width: 252px;
            background: var(--sidebar-bg);
            border-right: 1px solid var(--sidebar-border);
            display: flex;
            flex-direction: column;
            height: 100vh;
        }

        .sidebar-top {
            padding: 22px 18px 18px;
            border-bottom: 1px solid var(--sidebar-border);
        }
        .brand { display: flex; align-items: center; gap: 10px; }
        .brand-logo {
            width: 34px; height: 34px;
            background: var(--accent);
            border-radius: 9px;
            display: flex; align-items: center; justify-content: center;
            font-size: 17px;
            flex-shrink: 0;
            box-shadow: 0 2px 8px rgba(193,127,62,0.25);
        }
        .brand-name {
            font-family: 'Noto Serif SC', serif;
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
            line-height: 1.3;
        }
        .brand-sub {
            font-size: 10px;
            color: var(--text-muted);
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-top: 8px;
        }

        .sidebar-section { padding: 16px 14px 0; }
        .sidebar-label {
            font-size: 10px;
            color: var(--text-muted);
            letter-spacing: 1.2px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .file-btn {
            width: 100%;
            padding: 9px 12px;
            background: var(--white);
            border: 1.5px dashed var(--border);
            border-radius: 8px;
            color: var(--text-secondary);
            font-family: 'DM Sans', sans-serif;
            font-size: 12.5px;
            cursor: pointer;
            text-align: left;
            transition: all 0.18s;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .file-btn:hover { border-color: var(--accent-border); color: var(--text-primary); background: var(--accent-light); }
        .file-btn.selected { border-color: var(--accent); border-style: solid; background: var(--accent-light); color: var(--accent-dark); }
        .file-btn-text { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }

        .process-btn {
            width: 100%;
            margin-top: 8px;
            padding: 9px;
            background: var(--accent);
            border: none;
            border-radius: 8px;
            color: white;
            font-family: 'DM Sans', sans-serif;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.18s;
            box-shadow: 0 2px 6px rgba(193,127,62,0.2);
        }
        .process-btn:hover:not(:disabled) { background: var(--accent-dark); }
        .process-btn:disabled { opacity: 0.38; cursor: not-allowed; box-shadow: none; }

        .progress-box { padding: 10px 2px 0; display: none; }
        .progress-track { height: 3px; background: var(--border); border-radius: 2px; overflow: hidden; }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-dark), var(--accent));
            width: 0%;
            transition: width 0.5s ease;
        }
        .progress-txt { font-size: 11px; color: var(--text-muted); margin-top: 5px; font-family: 'DM Mono', monospace; }

        .sidebar-papers {
            flex: 1;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            padding: 16px 14px 0;
        }
        .paper-scroll { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 2px; }
        .paper-scroll::-webkit-scrollbar { width: 3px; }
        .paper-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

        .paper-item {
            padding: 8px 10px;
            border-radius: 7px;
            cursor: pointer;
            color: var(--text-secondary);
            font-size: 11.5px;
            font-family: 'DM Mono', monospace;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            transition: all 0.15s;
            border: 1px solid transparent;
        }
        .paper-item:hover { background: var(--white); color: var(--text-primary); border-color: var(--border); }
        .paper-item.active { background: var(--accent-light); color: var(--accent-dark); border-color: var(--accent-border); font-weight: 500; }
        .paper-empty { font-size: 12px; color: var(--text-muted); font-style: italic; padding: 2px; }

        .sidebar-footer {
            padding: 12px 14px 14px;
            border-top: 1px solid var(--sidebar-border);
            font-size: 10px;
            color: var(--text-muted);
            letter-spacing: 0.4px;
        }

        /* ══ 主区域 ══ */
        .main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

        .topbar {
            height: 54px; min-height: 54px;
            background: var(--white);
            border-bottom: 1px solid var(--border);
            display: flex; align-items: center; justify-content: space-between;
            padding: 0 32px;
            box-shadow: var(--shadow-sm);
        }
        .topbar-left { display: flex; align-items: center; gap: 10px; }
        .topbar-dot {
            width: 7px; height: 7px;
            border-radius: 50%;
            background: var(--border);
            transition: all 0.3s;
        }
        .topbar-dot.active { background: #22c55e; box-shadow: 0 0 0 3px rgba(34,197,94,0.15); }
        .topbar-title { font-size: 13.5px; color: var(--text-secondary); }
        .topbar-title strong { color: var(--text-primary); font-weight: 500; }

        .clear-btn {
            font-size: 12px; color: var(--text-muted);
            background: none; border: 1px solid var(--border);
            padding: 5px 14px; border-radius: 20px; cursor: pointer;
            transition: all 0.15s; font-family: 'DM Sans', sans-serif;
        }
        .clear-btn:hover { border-color: var(--text-secondary); color: var(--text-secondary); }

        /* 对话区 */
        .chat-area { flex: 1; overflow-y: auto; padding: 28px 0; scroll-behavior: smooth; }
        .chat-area::-webkit-scrollbar { width: 5px; }
        .chat-area::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
        .chat-inner { max-width: 700px; margin: 0 auto; padding: 0 20px; display: flex; flex-direction: column; gap: 18px; }

        /* 空状态 */
        .empty-state { max-width: 680px; margin: 60px auto 0; padding: 0 24px; text-align: center; }
        .empty-icon {
            width: 52px; height: 52px;
            background: var(--accent-light);
            border: 1px solid var(--accent-border);
            border-radius: 14px;
            display: flex; align-items: center; justify-content: center;
            font-size: 24px;
            margin: 0 auto 18px;
        }
        .empty-title { font-family: 'Noto Serif SC', serif; font-size: 21px; font-weight: 600; color: var(--text-primary); margin-bottom: 8px; }
        .empty-sub { font-size: 13.5px; color: var(--text-muted); line-height: 1.75; }
        .empty-tips { margin-top: 28px; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
        .tip-card {
            background: var(--white);
            border: 1px solid var(--border);
            border-radius: 10px; padding: 14px 13px; text-align: left;
            box-shadow: var(--shadow-sm);
            transition: box-shadow 0.2s;
        }
        .tip-card:hover { box-shadow: var(--shadow-md); }
        .tip-card-icon { font-size: 17px; margin-bottom: 7px; }
        .tip-card-title { font-size: 12px; font-weight: 500; color: var(--text-primary); margin-bottom: 3px; }
        .tip-card-desc { font-size: 11px; color: var(--text-muted); line-height: 1.55; }

        /* 消息 */
        .msg-row { animation: msgIn 0.22s ease; }
        @keyframes msgIn { from { opacity:0; transform:translateY(5px); } to { opacity:1; transform:translateY(0); } }

        .msg-user-wrap { display: flex; justify-content: flex-end; }
        .msg-user {
            background: var(--user-bg);
            border: 1px solid var(--user-border);
            color: #312e81;
            padding: 10px 15px;
            border-radius: 16px 16px 3px 16px;
            max-width: 70%;
            font-size: 14px; line-height: 1.65; white-space: pre-wrap;
        }

        .msg-bot-wrap { display: flex; align-items: flex-start; gap: 10px; }
        .bot-avatar {
            width: 28px; height: 28px; min-width: 28px;
            background: var(--accent);
            border-radius: 7px;
            display: flex; align-items: center; justify-content: center;
            font-family: 'Noto Serif SC', serif;
            font-size: 13px; color: white; font-weight: 600;
            margin-top: 2px;
            box-shadow: 0 2px 6px rgba(193,127,62,0.2);
        }
        .msg-bot {
            background: var(--white);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 12px 16px;
            border-radius: 3px 16px 16px 16px;
            max-width: 88%;
            font-size: 14px; line-height: 1.8; white-space: pre-wrap;
            box-shadow: var(--shadow-sm);
        }
        .msg-bot.error-msg { background: #fff5f5; border-color: #fca5a5; color: #b91c1c; }
        .msg-source {
            margin-left: 38px; margin-top: 4px;
            font-size: 11px; color: var(--accent);
            font-family: 'DM Mono', monospace;
            display: flex; align-items: center; gap: 5px;
        }
        .msg-source::before { content:''; display:inline-block; width:3px; height:3px; background:var(--accent); border-radius:50%; }

        .typing-cursor::after { content:'|'; animation: blink 0.7s step-end infinite; color: var(--accent); }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

        /* 输入区 */
        .input-area {
            background: var(--white);
            border-top: 1px solid var(--border);
            padding: 13px 20px 16px;
        }
        .input-inner { max-width: 700px; margin: 0 auto; }
        .input-box {
            display: flex; align-items: flex-end; gap: 10px;
            background: var(--main-bg);
            border: 1.5px solid var(--border);
            border-radius: 13px;
            padding: 9px 10px 9px 15px;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        .input-box:focus-within { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(193,127,62,0.09); }
        .input-box textarea {
            flex: 1; background: none; border: none; outline: none;
            color: var(--text-primary);
            font-family: 'DM Sans', sans-serif; font-size: 14px; font-weight: 400;
            resize: none; line-height: 1.6; min-height: 22px; max-height: 130px;
        }
        .input-box textarea::placeholder { color: var(--text-muted); }
        .send-btn {
            width: 34px; height: 34px;
            background: var(--accent); border: none; border-radius: 9px;
            cursor: pointer; display: flex; align-items: center; justify-content: center;
            font-size: 14px; color: white; flex-shrink: 0;
            transition: all 0.15s;
            box-shadow: 0 2px 5px rgba(193,127,62,0.25);
        }
        .send-btn:hover:not(:disabled) { background: var(--accent-dark); transform: scale(1.05); }
        .send-btn:disabled { opacity: 0.28; cursor: not-allowed; transform: none; box-shadow: none; }
        .input-hint { text-align: right; font-size: 11px; color: var(--text-muted); margin-top: 5px; }
    </style>
</head>
<body>

<div class="sidebar">
    <div class="sidebar-top">
        <div class="brand">
            <div class="brand-logo">📄</div>
            <div>
                <div class="brand-name">论文问答助手</div>
            </div>
        </div>
        <div class="brand-sub">RAG · HyDE · 多轮对话</div>
    </div>

    <div class="sidebar-section" style="padding-top:16px;">
        <div class="sidebar-label">上传论文</div>
        <button class="file-btn" id="fileBtn" onclick="document.getElementById('pdfFile').click()">
            <span>📂</span>
            <span class="file-btn-text" id="fileBtnText">选择 PDF 文件</span>
        </button>
        <input type="file" id="pdfFile" accept=".pdf" style="display:none" onchange="onFileSelect(this)">
        <button class="process-btn" id="processBtn" onclick="uploadPDF()" disabled>建立索引</button>
        <div class="progress-box" id="progressBox">
            <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
            <div class="progress-txt" id="progressTxt">准备中...</div>
        </div>
    </div>

    <div class="sidebar-papers">
        <div class="sidebar-label">已加载论文</div>
        <div class="paper-scroll" id="paperList">
            <div class="paper-empty">暂无论文</div>
        </div>
    </div>

    <div class="sidebar-footer">bge-m3 · Qwen2.5-72B · Faiss</div>
</div>

<div class="main">
    <div class="topbar">
        <div class="topbar-left">
            <div class="topbar-dot" id="topDot"></div>
            <div class="topbar-title" id="topTitle">选择一篇论文开始提问</div>
        </div>
        <button class="clear-btn" onclick="clearHistory()">清空对话</button>
    </div>

    <div class="chat-area" id="chatArea">
        <div class="empty-state" id="emptyState">
            <div class="empty-icon">🔬</div>
            <div class="empty-title">论文问答助手</div>
            <div class="empty-sub">上传学术 PDF，即可用中文提问<br>支持多文件管理与多轮追问</div>
            <div class="empty-tips">
                <div class="tip-card">
                    <div class="tip-card-icon">⚡</div>
                    <div class="tip-card-title">HyDE 增强检索</div>
                    <div class="tip-card-desc">生成假设答案再检索，精准匹配语义</div>
                </div>
                <div class="tip-card">
                    <div class="tip-card-icon">💬</div>
                    <div class="tip-card-title">多轮对话</div>
                    <div class="tip-card-desc">支持连续追问，保留上下文记忆</div>
                </div>
                <div class="tip-card">
                    <div class="tip-card-icon">📍</div>
                    <div class="tip-card-title">答案溯源</div>
                    <div class="tip-card-desc">每条回答标注原文参考页码</div>
                </div>
            </div>
        </div>
        <div class="chat-inner" id="chatInner" style="display:none;"></div>
    </div>

    <div class="input-area">
        <div class="input-inner">
            <div class="input-box">
                <textarea id="question" placeholder="输入问题，支持追问…" rows="1"
                    oninput="autoResize(this)" onkeydown="handleKey(event)"></textarea>
                <button class="send-btn" id="sendBtn" onclick="askQuestion()" disabled>➤</button>
            </div>
            <div class="input-hint">Enter 发送 · Shift+Enter 换行</div>
        </div>
    </div>
</div>

<script>
    function onFileSelect(input) {
        const file = input.files[0];
        if (!file) return;
        document.getElementById('fileBtn').classList.add('selected');
        document.getElementById('fileBtnText').innerText = file.name;
        document.getElementById('processBtn').disabled = false;
    }

    function autoResize(el) {
        el.style.height = 'auto';
        el.style.height = Math.min(el.scrollHeight, 130) + 'px';
    }

    function handleKey(e) {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); askQuestion(); }
    }

    async function uploadPDF() {
        const file = document.getElementById('pdfFile').files[0];
        if (!file) return;
        const processBtn = document.getElementById('processBtn');
        const progressBox = document.getElementById('progressBox');
        const progressFill = document.getElementById('progressFill');
        const progressTxt = document.getElementById('progressTxt');
        processBtn.disabled = true;
        progressBox.style.display = 'block';
        const stages = [['提取文本…','20%'],['切分文字块…','42%'],['生成向量嵌入…','65%'],['构建检索索引…','85%']];
        let si = 0;
        const timer = setInterval(() => {
            if (si < stages.length) { progressTxt.innerText = stages[si][0]; progressFill.style.width = stages[si][1]; si++; }
        }, 2800);
        const fd = new FormData();
        fd.append('file', file);
        try {
            const res = await fetch('/upload', { method: 'POST', body: fd });
            const data = await res.json();
            clearInterval(timer);
            if (data.error) {
                progressFill.style.background = '#fca5a5';
                progressFill.style.width = '100%';
                progressTxt.innerText = '✗ ' + data.error;
            } else {
                progressFill.style.width = '100%';
                progressTxt.innerText = '✓ ' + data.chunks + ' 个文字块';
                renderPaperList(data.paper_list, data.filename);
                document.getElementById('sendBtn').disabled = false;
                document.getElementById('topDot').classList.add('active');
                updateTopTitle(data.filename);
                showChat();
            }
        } catch(e) {
            clearInterval(timer);
            progressTxt.innerText = '✗ 网络错误';
            progressFill.style.background = '#fca5a5';
            progressFill.style.width = '100%';
        } finally {
            processBtn.disabled = false;
        }
    }

    function renderPaperList(list, current) {
        const el = document.getElementById('paperList');
        if (!list || list.length === 0) { el.innerHTML = '<div class="paper-empty">暂无论文</div>'; return; }
        el.innerHTML = list.map(name => {
            const s = name.length > 26 ? name.slice(0,26)+'…' : name;
            return `<div class="paper-item ${name===current?'active':''}" onclick="switchPaper('${name}')" title="${name}">${s}</div>`;
        }).join('');
    }

    async function switchPaper(filename) {
        const res = await fetch('/switch', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({filename}) });
        const data = await res.json();
        if (data.message) {
            const r = await fetch('/papers');
            const d = await r.json();
            renderPaperList(d.paper_list, d.current);
            updateTopTitle(filename);
            document.getElementById('chatInner').innerHTML = '';
            showChat();
            document.getElementById('sendBtn').disabled = false;
        }
    }

    function updateTopTitle(name) {
        const s = name.length > 48 ? name.slice(0,48)+'…' : name;
        document.getElementById('topTitle').innerHTML = '<strong>' + s + '</strong>';
    }

    function showChat() {
        const es = document.getElementById('emptyState');
        if (es) es.style.display = 'none';
        document.getElementById('chatInner').style.display = 'flex';
    }

    function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
    function scrollBottom() { const a = document.getElementById('chatArea'); a.scrollTop = a.scrollHeight; }

    function appendUserMsg(text) {
        const inner = document.getElementById('chatInner');
        const row = document.createElement('div');
        row.className = 'msg-row';
        row.innerHTML = `<div class="msg-user-wrap"><div class="msg-user">${escHtml(text)}</div></div>`;
        inner.appendChild(row);
        scrollBottom();
    }

    function appendBotPlaceholder() {
        const inner = document.getElementById('chatInner');
        const row = document.createElement('div');
        row.className = 'msg-row';
        const uid = 'bot_' + Date.now();
        row.innerHTML = `<div class="msg-bot-wrap"><div class="bot-avatar">A</div><div class="msg-bot typing-cursor" id="${uid}"></div></div>`;
        inner.appendChild(row);
        scrollBottom();
        return { row, uid };
    }

    function typeText(el, text, onDone) {
        el.classList.add('typing-cursor');
        let idx = 0;
        const spd = Math.max(6, Math.min(18, 1800 / text.length));
        const t = setInterval(() => {
            el.innerText = text.slice(0, ++idx);
            scrollBottom();
            if (idx >= text.length) { clearInterval(t); el.classList.remove('typing-cursor'); if (onDone) onDone(); }
        }, spd);
    }

    async function askQuestion() {
        const input = document.getElementById('question');
        const q = input.value.trim();
        if (!q) return;
        appendUserMsg(q);
        input.value = ''; input.style.height = 'auto';
        const sendBtn = document.getElementById('sendBtn');
        sendBtn.disabled = true;
        const { row, uid } = appendBotPlaceholder();
        const botEl = document.getElementById(uid);
        botEl.innerText = '思考中...';
        try {
            const res = await fetch('/ask', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question:q}) });
            const data = await res.json();
            if (data.answer && data.answer.startsWith('⚠️')) {
                botEl.classList.remove('typing-cursor');
                botEl.classList.add('error-msg');
                botEl.innerText = data.answer;
            } else {
                typeText(botEl, data.answer, () => {
                    if (data.sources && data.sources.length > 0) {
                        const src = document.createElement('div');
                        src.className = 'msg-source';
                        src.innerText = '参考页码：第 ' + data.sources.sort((a,b)=>a-b).join('、') + ' 页';
                        row.appendChild(src);
                        scrollBottom();
                    }
                });
            }
        } catch(e) {
            botEl.classList.remove('typing-cursor');
            botEl.classList.add('error-msg');
            botEl.innerText = '网络错误，请检查 Flask 服务是否运行';
        } finally {
            sendBtn.disabled = false;
            input.focus();
        }
    }

    async function clearHistory() {
        await fetch('/clear_history', {method:'POST'});
        document.getElementById('chatInner').innerHTML = '';
    }
</script>
</body>
</html>
"""


@app.route('/')
def home():
    return render_template_string(HTML)


if __name__ == '__main__':
    app.run(debug=True, port=5000)