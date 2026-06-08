import os
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


# =========================================================
# 1. 定义输出结构（Output Parser 的 Schema）
# =========================================================

class DocAnalysisOutput(BaseModel):
    """文档分析的结构化输出"""
    summary: str = Field(description="文档的核心内容摘要，150-200字")
    keywords: List[str] = Field(description="5-10个关键词")
    sentiment: str = Field(description="情感倾向：positive/negative/neutral")
    sentiment_reason: str = Field(description="判断情感的原因")
    document_type: str = Field(description="文档类型：合同/论文/新闻/报告/简历等")
    key_points: List[str] = Field(description="3-5个核心要点")
    readability_score: int = Field(description="可读性评分，1-10分")
    suggestions: List[str] = Field(description="阅读建议")


# 创建输出解析器
output_parser = PydanticOutputParser(pydantic_object=DocAnalysisOutput)


# =========================================================
# 2. 创建大模型
# =========================================================

def create_llm(temperature: float = 0.3):
    """创建 DeepSeek 大模型"""
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

    return ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=temperature,
    )


# =========================================================
# 3. 文档分析助手类
# =========================================================

class DocAnalyzer:
    """
    智能文档分析助手
    技术点：
    1. PromptTemplate - 通过 ChatPromptTemplate 实现
    2. OutputParser - 通过 PydanticOutputParser 实现
    3. Chain - 通过 LCEL (|) 实现
    4. Memory - 通过手动管理消息列表实现
    """

    def __init__(self, llm):
        self.llm = llm

        # Memory：存储对话历史
        self.chat_history: List[Dict] = []

        # 当前文档信息
        self.current_doc_content = ""
        self.current_doc_name = ""

        # 创建 Chain（使用 LCEL）
        self.analysis_chain = self._build_analysis_chain()
        self.qa_chain = self._build_qa_chain()

    def _build_analysis_chain(self):
        """构建文档分析 Chain（PromptTemplate + OutputParser）"""

        # Prompt Template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位专业的文档分析专家。

请分析用户提供的文档，并严格按照以下格式输出：

{format_instructions}

分析要求：
1. 摘要要准确概括核心内容
2. 关键词要能代表文档主题
3. 情感分析要客观（positive/negative/neutral）
4. 可读性评分 1-10 分
5. 建议要具体有用"""),
            ("human", "请分析以下文档：\n\n{document_content}")
        ])

        # Chain：prompt -> llm -> output_parser
        return prompt | self.llm | output_parser

    def _build_qa_chain(self):
        """构建问答 Chain（PromptTemplate + Memory）"""

        # Prompt Template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是智能文档分析助手。请基于当前文档回答用户的问题。

当前文档信息：
- 文件名：{doc_name}
- 文档内容预览：{doc_preview}

回答要求：
1. 只基于文档内容回答，不要编造信息
2. 如果问题超出文档范围，明确说明
3. 回答要准确、简洁
4. 可以引用文档中的具体内容"""),
            ("human", "{question}")
        ])

        # Chain：prompt -> llm
        return prompt | self.llm

    def analyze_document(self, doc_content: str, doc_name: str) -> Dict[str, Any]:
        """
        分析文档
        使用 Chain 执行
        """
        # 保存文档信息
        self.current_doc_content = doc_content
        self.current_doc_name = doc_name

        # 清空之前的记忆
        self.chat_history = []

        try:
            # 调用 Chain 执行分析
            result = self.analysis_chain.invoke({
                "format_instructions": output_parser.get_format_instructions(),
                "document_content": doc_content
            })

            # 保存到记忆
            self.chat_history.append({
                "role": "user",
                "content": f"用户上传了文档：{doc_name}"
            })
            self.chat_history.append({
                "role": "assistant",
                "content": f"已完成文档分析，摘要：{result.summary[:100]}..."
            })

            return {
                "success": True,
                "data": result.model_dump(),
                "raw_output": str(result)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def ask_question(self, question: str) -> str:
        if not self.current_doc_content:
            return "请先上传并分析一份文档。"

        try:
            #文档预览
            doc_preview = self.current_doc_content[:2000]
            if len(self.current_doc_content) > 2000:
                doc_preview += "..."

            #调用 Chain
            result = self.qa_chain.invoke({
                "doc_name": self.current_doc_name,
                "doc_preview": doc_preview,
                "question": question
            })

            # 获取回答内容
            response = result.content if hasattr(result, 'content') else str(result)

            # 保存到记忆
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": response})

            return response

        except Exception as e:
            return f"处理问题时出错：{e}"

    def get_memory_summary(self) -> str:
        """获取对话记忆摘要"""
        if not self.chat_history:
            return "暂无对话记录"

        summary = f"对话记录（共 {len(self.chat_history)} 条）：\n"
        for msg in self.chat_history[-6:]:
            role = "用户" if msg["role"] == "user" else "助手"
            content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
            summary += f"  • {role}: {content}\n"

        return summary

    def clear_memory(self):
        """清空记忆"""
        self.chat_history = []


# =========================================================
# 4. 工厂函数
# =========================================================

def create_analyzer(temperature: float = 0.3):
    #创建文档分析器
    llm = create_llm(temperature=temperature)
    return DocAnalyzer(llm)