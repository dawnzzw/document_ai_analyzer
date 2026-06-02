import streamlit as st
import os
from datetime import datetime
import config
import utils

# 页面配置
st.set_page_config(
    page_title="📄 智能文档分析系统",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化目录
utils.ensure_directories()

# 自定义 CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.2em;
    }
    </style>
""", unsafe_allow_html=True)

# 标题
st.title("📄 智能文档分析系统")
st.markdown("---")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 应用设置")

    # API 密钥检查
    if not config.OPENAI_API_KEY:
        st.error("❌ 未检测到 OpenAI API Key")
        st.info("请在 `.env` 文件中设置 `OPENAI_API_KEY`")
    else:
        st.success("✅ OpenAI 连接正常")

    st.markdown("---")
    st.write(f"**配置信息:**")
    st.write(f"- 模型: {config.LLM_MODEL}")
    st.write(f"- Embedding: {config.EMBEDDING_MODEL}")
    st.write(f"- Chunk大小: {config.CHUNK_SIZE}")

# 主界面 - 标签页
tab1, tab2, tab3 = st.tabs(["📤 上传新文档", "📚 已有知识库", "❓ 智能问答"])

# ==================== 标签页1：上传新文档 ====================
with tab1:
    st.header("📤 上传并分析文档")

    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "选择文档文件",
            type=["pdf", "docx", "txt", "md"],
            help="支持 PDF、Word、文本和 Markdown 文件"
        )

    with col2:
        st.write(f"**限制**: {config.MAX_FILE_SIZE // (1024 * 1024)}MB")

    if uploaded_file is not None:
        # 检查文件大小
        file_size_mb = utils.get_file_size_mb(uploaded_file.size)
        if file_size_mb > config.MAX_FILE_SIZE / (1024 * 1024):
            st.error(f"❌ 文件过大 ({file_size_mb:.2f}MB)")
        else:
            st.success(f"✅ 文件大小: {file_size_mb:.2f}MB")

            # 获取向量库名称
            store_name = uploaded_file.name.replace(".", "_").replace(" ", "_")

            if st.button("🚀 开始处理文档", use_container_width=True):
                try:
                    with st.spinner("⏳ 正在处理文档..."):
                        # 1. 保存文件
                        file_path = utils.save_uploaded_file(uploaded_file, uploaded_file.name)
                        st.info(f"✓ 文件已保存")

                        # 2. 加载文档
                        documents = utils.load_document(file_path)
                        st.info(f"✓ 文档已加载 ({len(documents)} 页)")

                        # 3. 分割文档
                        split_docs = utils.split_documents(documents)
                        st.info(f"✓ 文档已分割 ({len(split_docs)} 块)")

                        # 4. 创建向量库
                        vector_store_path = utils.create_vector_store(split_docs, store_name)
                        st.success(f"✅ 向量库已创建！")

                        # 显示统计信息
                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("文档页数", len(documents))
                        with col2:
                            st.metric("文本块数", len(split_docs))
                        with col3:
                            st.metric("创建时间", datetime.now().strftime("%H:%M:%S"))

                        # 保存到 session
                        st.session_state.current_store = store_name
                        st.session_state.doc_name = uploaded_file.name

                except Exception as e:
                    st.error(f"❌ 处理失败: {str(e)}")

# ==================== 标签页2：已有知识库 ====================
with tab2:
    st.header("📚 已有知识库")

    stored_stores = utils.get_stored_vector_stores()

    if not stored_stores:
        st.info("📭 暂无已保存的知识库，请先上传文档")
    else:
        st.write(f"**已保存的知识库 ({len(stored_stores)} 个):**")

        for store_name in stored_stores:
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                if st.button(f"📖 {store_name}", use_container_width=True):
                    st.session_state.current_store = store_name
                    st.switch_page("pages/01_📄_文档分析.py")

            with col2:
                if st.button("🔍", key=f"view_{store_name}"):
                    st.session_state.current_store = store_name

            with col3:
                if st.button("🗑️", key=f"delete_{store_name}"):
                    utils.delete_vector_store(store_name)
                    st.rerun()

# ==================== 标签页3：智能问答 ====================
with tab3:
    st.header("❓ 智能问答")

    col1, col2 = st.columns([3, 1])

    with col1:
        # 选择知识库
        available_stores = utils.get_stored_vector_stores()

        if not available_stores:
            st.warning("⚠️ 请先上传文档创建知识库")
        else:
            selected_store = st.selectbox(
                "选择知识库",
                available_stores,
                key="kb_select"
            )

            with col2:
                st.write("")
                if st.button("🔄 刷新", use_container_width=True):
                    st.rerun()

            st.markdown("---")

            # 问答界面
            user_question = st.text_area(
                "请输入您的问题：",
                placeholder="例如：这份文档的主要内容是什么？",
                height=100
            )

            col1, col2 = st.columns([1, 1])

            with col1:
                if st.button("💭 提交问题", use_container_width=True):
                    if not user_question.strip():
                        st.error("❌ 请输入问题")
                    else:
                        try:
                            with st.spinner("🤔 AI 思考中..."):
                                from langchain.chains import RetrievalQA
                                from langchain.chat_models import ChatOpenAI

                                # 加载向量库
                                vectorstore = utils.load_vector_store(selected_store)

                                # 初始化 LLM
                                llm = ChatOpenAI(
                                    model_name=config.LLM_MODEL,
                                    temperature=config.TEMPERATURE,
                                    max_tokens=config.MAX_TOKENS
                                )

                                # 创建 QA 链
                                qa_chain = RetrievalQA.from_chain_type(
                                    llm=llm,
                                    chain_type="stuff",
                                    retriever=vectorstore.as_retriever(
                                        search_kwargs={"k": config.TOP_K_RETRIEVAL}
                                    ),
                                    return_source_documents=True
                                )

                                # 执行查询
                                result = qa_chain({"query": user_question})

                            # 显示回答
                            st.markdown("---")
                            st.subheader("📌 AI 回答")
                            st.markdown(result["result"])

                            # 显示来源
                            st.markdown("---")
                            st.subheader("📚 参考来源")

                            for i, doc in enumerate(result["source_documents"], 1):
                                page_num = doc.metadata.get("page", "N/A")
                                with st.expander(f"📄 来源 {i}（第 {page_num} 页）"):
                                    st.write(doc.page_content)

                        except Exception as e:
                            st.error(f"❌ 查询失败: {str(e)}")

            with col2:
                if st.button("🗑️ 清空", use_container_width=True):
                    st.session_state.clear()
                    st.rerun()

# ==================== 页脚 ====================
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9em;">
        <p>📄 智能文档分析系统 | Powered by LangChain + Streamlit + OpenAI</p>
        <p>❤️ 提问、分析、洞察</p>
    </div>
""", unsafe_allow_html=True)