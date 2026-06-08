"""
app.py - 智能文档分析助手界面
"""
import os
import streamlit as st
from doc_core import create_analyzer

# 页面配置
st.set_page_config(
    page_title="智能文档分析助手",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# 初始化
# =========================================================

def init_session_state():
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "doc_loaded" not in st.session_state:
        st.session_state.doc_loaded = False


init_session_state()


# =========================================================
# 文件读取函数
# =========================================================

def read_file(uploaded_file) -> tuple:
    """读取上传的文件，返回 (内容, 文件名)"""
    file_name = uploaded_file.name
    file_type = file_name.split(".")[-1].lower()

    content = ""

    if file_type == "txt":
        content = uploaded_file.read().decode("utf-8")

    elif file_type == "pdf":
        try:
            import pypdf
            pdf_reader = pypdf.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                content += page.extract_text()
        except ImportError:
            st.error("请先安装 pypdf：pip install pypdf")
            return "", ""

    elif file_type == "docx":
        try:
            import docx
            doc = docx.Document(uploaded_file)
            content = "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            st.error("请先安装 python-docx：pip install python-docx")
            return "", ""

    else:
        st.error(f"不支持的文件类型：{file_type}")
        return "", ""

    return content, file_name


# =========================================================
# 侧边栏
# =========================================================

def setup_sidebar():
    st.sidebar.title("⚙️ 设置")

    api_key = st.sidebar.text_input(
        "DeepSeek API Key",
        type="password",
        placeholder="留空则读取环境变量"
    )

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="越低越稳定，越高越有创意"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📖 使用说明")
    st.sidebar.markdown("""
    1. 上传文档（txt/pdf/docx）
    2. 点击「开始分析」
    3. 查看摘要、关键词、情感分析
    4. 在对话框提问
    """)

    st.sidebar.markdown("---")

    if st.sidebar.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.analysis_result = None
        st.session_state.doc_loaded = False
        if st.session_state.analyzer:
            st.session_state.analyzer.clear_memory()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 LangChain 技术点")
    st.sidebar.markdown("""
    - **PromptTemplate**: `ChatPromptTemplate`
    - **OutputParser**: `PydanticOutputParser`
    - **Chain**: `LCEL` (`prompt | llm | parser`)
    - **Memory**: 手动管理对话历史
    """)

    return api_key, temperature


# =========================================================
# 文档上传区
# =========================================================

def document_upload_section():
    """文档上传区域"""
    st.subheader("📂 上传文档")

    uploaded_file = st.file_uploader(
        "选择文档文件",
        type=["txt", "pdf", "docx"],
        help="支持 .txt、.pdf、.docx 格式"
    )

    if uploaded_file is not None:
        try:
            content, name = read_file(uploaded_file)
            if content:
                st.success(f"✅ 已上传：{name} ({len(content)} 字符)")

                with st.expander("📄 文档预览"):
                    st.text(content[:1000] + ("..." if len(content) > 1000 else ""))

                return content, name
            else:
                return "", ""
        except Exception as e:
            st.error(f"读取文件失败：{e}")
            return "", ""

    return "", ""


# =========================================================
# 分析结果展示
# =========================================================

def display_analysis(result: dict):
    """展示分析结果"""
    st.subheader("📊 分析报告")

    if not result.get("success"):
        st.error(f"分析失败：{result.get('error', '未知错误')}")
        return

    data = result.get("data")
    if not data:
        st.warning("分析结果解析失败")
        return

    # 三列指标
    col1, col2, col3 = st.columns(3)

    with col1:
        sentiment = data.get("sentiment", "unknown")
        sentiment_map = {
            "positive": "😊 正面",
            "negative": "😞 负面",
            "neutral": "😐 中性"
        }
        st.metric("情感倾向", sentiment_map.get(sentiment, sentiment))

    with col2:
        score = data.get("readability_score", 0)
        st.metric("可读性评分", f"{score}/10")
        st.progress(score / 10)

    with col3:
        doc_type = data.get("document_type", "未知")
        st.metric("文档类型", doc_type)

    # 摘要
    with st.expander("📝 摘要", expanded=True):
        st.write(data.get("summary", "暂无"))

    # 关键词
    with st.expander("🔑 关键词", expanded=True):
        keywords = data.get("keywords", [])
        st.markdown(" ".join([f"`{kw}`" for kw in keywords]))

    # 核心要点
    with st.expander("🎯 核心要点", expanded=True):
        for point in data.get("key_points", []):
            st.markdown(f"- {point}")

    # 建议
    with st.expander("💡 建议", expanded=False):
        suggestions = data.get("suggestions", [])
        for sug in suggestions:
            st.markdown(f"- {sug}")

    # 情感原因
    with st.expander("📖 情感分析依据", expanded=False):
        st.write(data.get("sentiment_reason", "暂无"))


# =========================================================
# 聊天区
# =========================================================

def chat_section(analyzer):
    """对话区域"""
    st.subheader("💬 文档问答")
    st.caption("基于文档内容提问，AI 会记住对话历史")

    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用户输入
    if prompt := st.chat_input("例如：这篇文档的核心观点是什么？"):
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        # AI 回复
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = analyzer.ask_question(prompt)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


# =========================================================
# 主函数
# =========================================================

def main():
    st.title("📄 智能文档分析助手")
    st.caption("基于 LangChain 框架 | 支持文档摘要、关键词提取、情感分析、智能问答")

    # 侧边栏
    api_key, temperature = setup_sidebar()

    # 设置 API Key
    if api_key:
        os.environ["DEEPSEEK_API_KEY"] = api_key

    # 初始化分析器
    try:
        if st.session_state.analyzer is None:
            st.session_state.analyzer = create_analyzer(temperature=temperature)
        analyzer = st.session_state.analyzer
    except Exception as e:
        st.error(f"初始化失败：{e}")
        st.info("请检查 API Key 配置")
        return

    # 文档上传
    doc_content, doc_name = document_upload_section()

    # 分析按钮
    if doc_content and not st.session_state.doc_loaded:
        if st.button("🚀 开始分析文档", type="primary", use_container_width=True):
            with st.spinner("分析中..."):
                result = analyzer.analyze_document(doc_content, doc_name)
                st.session_state.analysis_result = result
                st.session_state.doc_loaded = True
                st.rerun()

    # 展示结果
    if st.session_state.analysis_result:
        display_analysis(st.session_state.analysis_result)

    # 聊天区
    if st.session_state.doc_loaded:
        chat_section(analyzer)

    # 显示 Memory 状态
    with st.sidebar.expander("💬 Memory 状态"):
        if analyzer:
            st.text(analyzer.get_memory_summary())


if __name__ == "__main__":
    main()