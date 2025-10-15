# LangChain Introducion

## 1. 介绍LangChain
### 1.1 什么是LangChain
LangChain是 2022年10月 ，由哈佛大学的 Harrison Chase （哈里森·蔡斯）发起研发的一个开源框架，用于开发由大语言模型（LLMs）驱动的应用程序。

比如，搭建“智能体”（Agent）、问答系统（QA）、对话机器人、文档搜索系统、企业私有知识库等。

![alt text](/public/langchain/intro/1.png)

### 1.2 有哪些大模型应用开发框架呢？

截止到2025年7月26日，GitHub统计数据：

![alt text](/public/langchain/intro/2.png)

- LangChain ：这些工具里出现最早、最成熟的，适合复杂任务分解和单智能体应用
- LlamaIndex ：专注于高效的索引和检索，适合 RAG 场景。（注意不是Meta开发的）
- LangChain4J ：LangChain还出了Java、JavaScript（LangChain.js）两个语言的版本，LangChain4j的功能略少于LangChain，但是主要的核心功能都是有的
- SpringAI/SpringAI Alibaba ：有待进一步成熟，此外只是简单的对于一些接口进行了封装
- SemanticKernel ：也称为sk，微软推出的，对于C#同学来说，那就是5颗星

### 1.3 为什么需要LangChain？

**问题1：LLMs用的好好的，干嘛还需要LangChain？**

在大语言模型（LLM）如 ChatGPT、Claude、DeepSeek 等快速发展的今天，开发者不仅希望能“使
用”这些模型，还希望能 `将它们灵活集成到自己的应用中` ，实现更强大的对话能力、检索增强生成（RAG）、工具调用（Tool Calling）、多轮推理等功能。

![alt text](/public/langchain/intro/3.png)
LangChain 为更方便解决这些问题，而生的。比如：大模型默认不能联网，如果需要联网，用
langchain。

**问题2：我们可以使用GPT 或GLM4 等模型的API进行开发，为何需要LangChain这样的框架？**

不使用LangChain，确实可以使用GPT 或GLM4 等模型的API进行开发。比如，搭建“智能体”
（Agent）、问答系统、对话机器人等复杂的 LLM 应用。

但使用LangChain的好处：

- **简化开发难度**：更简单、更高效、效果更好
- **学习成本更低**：不同模型的API不同，调用方式也有区别，切换模型时学习成本高。使用LangChain，可以以统一、规范的方式进行调用，有更好的移植性。
- **现成的链式组装**：LangChain提供了一些 `现成的链式组装` ，用于完成特定的高级任务。让复杂的逻辑变得 `结构化、易组合、易扩展`

**问题3：LangChain 提供了哪些功能呢？**

LangChain 是一个帮助你构建 LLM 应用的 `全套工具集` 。这里涉及到prompt 构建、LLM 接入、记忆管理、工具调用、RAG、智能体开发等模块。

### 1.4 LangChain的使用场景

学完LangChain，如下类型的项目，大家都可以实现：
|项目名称 |技术点 |难度|
|:---:|:---:|:---:|
|文档问答助手 | Prompt + Embedding + RetrievalQA |⭐⭐|
|智能日程规划助手| Agent + Tool + Memory |⭐⭐⭐|
|LLM+数据库问答 |SQLDatabaseToolkit + Agent |⭐⭐⭐⭐|
|多模型路由对话系统| RouterChain + 多 LLM |⭐⭐⭐⭐|
|互联网智能客服 |ConversationChain + RAG +Agent |⭐⭐⭐⭐⭐|
|企业知识库助手（RAG + 本地模型）|VectorDB + LLM + Streamlit |⭐⭐⭐⭐⭐|

LangChain的位置：

![alt text](/public/langchain/intro/4.png)

### 1.5 LangChain资料介绍
官网地址：https://www.langchain.com/langchain

官网文档：https://python.langchain.com/docs/introduction/

API文档：https://python.langchain.com/api_reference/

github地址：https://github.com/langchain-ai/langchain

### 1.6 架构设计
#### 1.6.1 总体架构图

**V0.1 版本**

![alt text](/public/langchain/intro/5.png)

**V0.2 / V0.3 版本**

![alt text](/public/langchain/intro/6.png)

图中展示了LangChain生态系统的主要组件及其分类，分为三个层次：架构(Architecture)、组件(Components)和部署(Deployment)。

> 版本的升级，v0.2 相较于v0.1，修改了⼤概10%-15%。功能性上差不多，主要是往稳定性（或兼容性）、安全性上使劲了，⽀持更多的⼤模型，更安全。

#### 1.6.2 内部架构详情

**结构1：LangChain**

**langchain**：构成应用程序认知架构的Chains，Agents，Retrieval strategies等
> 构成应⽤程序的链、智能体、RAG。

**langchain-community：第三方集成**

> ⽐如：Model I/O、Retrieval、Tool & Toolkit；合作伙伴包 langchain-openai，langchain-anthropic等。

**langchain-Core**：基础抽象和LangChain表达式语言 (LCEL)

小结：LangChain，就是AI应用组装套件，封装了一堆的API。langchain框架不大，但是里面琐碎的知识点特别多。就像玩乐高，提供了很多标准化的乐高零件（比如，连接器、轮子等）

**结构2：LangGraph**
LangGraph可以看做基于LangChain的api的进一步封装，能够协调多个Chain、Agent、Tools完成更复杂的任务，实现更高级的功能。


**结构3：LangSmith**

https://docs.smith.langchain.com/

**链路追踪**：提供了6大功能，涉及Debugging (调试)、Playground (沙盒)、Prompt Management (提示管理)、Annotation (注释)、Testing (测试)、Monitoring (监控)等。与LangChain无缝集成，帮助你从原型阶段过渡到生产阶段。

> 正是因为LangSmith这样的工具出现，才使得LangChain意义更大，更不仅靠一些API（当然也可以不用，用原生的API），支持不住LangChain的热度。

**结构4：LangServe**

将LangChain的可运行项和链部署为REST API，使得它们可以通过网络进行调用。
Java怎么调用langchain呢？就通过这个langserve。将langchain应用包装成一个rest api，对外暴露服务。同时，支持更高的并发，稳定性更好。

**总结：LangChain当中，最有前途的两个模块就是：LangGraph，LangSmith。**
> LangChain能做RAG，其他的一些框架也能做，而且做的不错，比如LlamaIndex。所以这时候LangChain要在Agent这块发力，那就需要LangGraph。而LangSmith，做运维、监控。故，二者是LangChain前最有前途的。

## 3. 大模型应用开发
大模型应用技术特点：门槛低，天花板高。

### 3.1 基于RAG架构的开发

**背景：**

- 大模型的知识冻结
- 大模型幻觉

而RAG就可以非常精准的解决这两个问题。

**举例：**

LLM在考试的时候面对陌生的领域，答复能力有限，然后就准备放飞自我了。而此时RAG给了一些提示和思路，让LLM懂了开始往这个提示的方向做，最终考试的正确率从60%到了90%！

![alt text](/public/langchain/intro/7.png)

**何为RAG？**

Retrieval-Augmented Generation（检索增强生成）

![alt text](/public/langchain/intro/8.png)

> 检索-增强-⽣成过程：检索可以理解为第10步，增强理解为第12步（这⾥的提⽰词包含检索到的数据），⽣成理解为第15步。

类似的细节图：

![alt text](/public/langchain/intro/9.png)

这些过程中的难点：1、文件解析 2、文件切割 3、知识检索 4、知识重排序

**Reranker的使用场景：**

- 适合：追求 回答高精度 和 高相关性 的场景中特别适合使用 Reranker，例如专业知识库或者客服系统等应用。

- 不适合：引入reranker会增加召回时间，增加检索延迟。服务对 响应时间要求高 时，使用reranker可能不合适。

**这里有三个位置涉及到大模型的使用：**

第3步向量化时，需要使用EmbeddingModels。

第7步重排序时，需要使用RerankModels。

第9步生成答案时，需要使用LLM。

### 3.2 基于Agent架构的开发
充分利用 LLM 的推理决策能力，通过增加 规划 、 记忆 和 工具 调用的能力，构造一个能够独立思考、逐步完成给定目标的智能体。

举例：传统的程序 vs Agent（智能体）

![alt text](/public/langchain/intro/10.png)

OpenAI的元老翁丽莲(Lilian Weng)于2023年6月在个人博客首次提出了现代AI Agent架构。

![alt text](/public/langchain/intro/11.png)

一个数学公式来表示：

**Agent = LLM + Memory + Tools + Planning + Action**

智能体核心要素被细化为一下模块：
1、大模型（LLM）作为“大脑”：提供推理、规划和知识理解能力，是AI Agent的决策中枢。

2、记忆（Memory）
- 短期记忆：存储单词对话周期的上下文信息，属于临时信息存储机制。受限于模型的上下文窗口长度。
- 长期记忆：可以横跨多个任务或时间周期，可存储并调用核心知识，非即时任务。
  - 长期记忆：可以通过模型参数微调（固化知识）、知识图谱（结构化语义网络）或向量数据库（相似性检索）方式实现。

3、工具使用（Tool Use）：调用外部工具（如API、数据库）扩展能力边界。

4、规划决策（Planning）：通过任务分解、反思与自省框架实现复杂任务处理。例如，利用思维链（Chain of Thought）将目标拆解为自任务，并通过反馈优化策略。

5、行动（Action）：实际执行决策的模块，涵盖软件接口操作（如自动订票）和物理交互（如机器人执行搬运）。比如：检索、推理、编程等。

### 3.3 大模型应用开发的4个场景

**场景1：纯 Prompt**

![alt text](/public/langchain/intro/12.png)

**场景2：Agent + Function Calling**

![alt text](/public/langchain/intro/13.png)

**场景3：RAG (Retrieval-Augmented Generation)**

![alt text](/public/langchain/intro/14.png)

**场景4：Fine-tuning(精调/微调)**

![alt text](/public/langchain/intro/15.png)

**如何选择**

面对一个需求，如何开始，如何选择技术方案？下面是个常用思路：

![alt text](/public/langchain/intro/16.png)

下面，我们重点介绍下大模型应用的开发两类：基于RAG的架构，基于Agent的架构。

## 4. LangChain的核心组件

学习Langchain最简单直接的方法就是阅读官方文档。

https://python.langchain.com/v0.1/docs/modules/

通过文档目录我们可以看到，Langchain构成的核心组件。

### 4.1 一个问题引发的思考

**如果要组织一个AI应用，开发者一般需要什么？**

第1，提示词模板的构建，不仅仅只包含用户输入。

第2，模型调用与返回，参数设置，返回内容的格式化输出。

第3，知识库查询，这里会包含文档加载，切割，以及转化为词嵌入（Embedding）向量。

第4，其他第三方工具调用，一般包含天气查询、Google搜索、一些自定义的接口能力调用。

第5，记忆获取，每一个对话都有上下文，在开启对话之前总得获取到之前的上下文吧？

**4.2 核心组件的概述**

LangChain的核心组件涉及六大模块，这六大模块提供了一个全面且强大的框架，使开发者能够创建复杂、高效且用户友好的基于大模型的应用。

![alt text](/public/langchain/intro/17.png)

### 4.3 核心组件的说明

**核心组件1：Model I/O**

Model I/O：标准化各个大模型的输入和输出，包含输入模版，模型本身和格式化输出。
以下是使用语言模型从输入到输出的基本流程。

![alt text](/public/langchain/intro/18.png)

以下是对每一块的总结：
- **Format**(格式化) ：即指代Prompts Template，通过模板管理大模型的输入。将原始数据格式化成模型可以处理的形式，插入到一个模板问题中，然后送入模型进行处理。
- **Predict**(预测) ：即指代Models，使用通用接口调用不同的大语言模型。接受被送进来的问题，然后基于这个问题进行预测或生成回答。
- **Parse**(生成) ：即指代Output Parser 部分，用来从模型的推理中提取信息，并按照预先设定好的模版来规范化输出。比如，格式化成一个结构化的JSON对象。

**核心组件2：Chains**

Chain："链条"，用于将多个模块串联起来组成一个完整的流程，是 LangChain 框架中最重要的模块。

例如，一个 Chain 可能包括一个 Prompt 模板、一个语言模型和一个输出解析器，它们一起工作以处理用户输入、生成响应并处理输出。

**常见的Chain类型：**

- **LLMChain** ：最基础的模型调用链
- **SequentialChain** ：多个链串联执行
- **RouterChain** ：自动分析用户的需求，引导到最适合的链
- **RetrievalQA** ：结合向量数据库进行问答的链

**核心组件3：Memory**

Memory：记忆模块，用于保存对话历史或上下文信息，以便在后续对话中使用。

**常见的 Memory 类型：**

- **ConversationBufferMemory** ：保存完整的对话历史
- **ConversationSummaryMemory** ：保存对话内容的精简摘要（适合长对话）
- **ConversationSummaryBufferMemory** ：混合型记忆机制，兼具上面两个类型的特点
- **VectorStoreRetrieverMemory** ：保存对话历史存储在向量数据库中

**核心组件4：Agents**

Agents，对应着智能体，是 LangChain 的高阶能力，它可以自主选择工具并规划执行步骤。

**Agent 的关键组成：**

- AgentType ：定义决策逻辑的工作流模式
- Tool ：是一些内置的功能模块，如API调用、搜索引擎、文本处理、数据查询等工具。Agents通过这些工具来执行特定的功能。
- AgentExecutor ：用来运行智能体并执行其决策的工具，负责协调智能体的决策和实际的工具执行。

**核心组件5：Retrieval**

Retrieval：对应着RAG，检索外部数据，然后在执行生成步骤时将其传递到 LLM。步骤包括文档加载、切割、Embedding等

![alt text](/public/langchain/intro/19.png)

- **Source** ：数据源，即大模型可以识别的多种类型的数据：视频、图片、文本、代码、文档等。
- **Load** ：负责将来自不同数据源的非结构化数据，加载为文档(Document)对象
- **Transform** ：负责对加载的文档进行转换和处理，比如将文本拆分为具有语义意义的小块。
- **Embed** ：将文本编码为向量的能力。一种用于嵌入文档，另一种用于嵌入查询
- **Store** ：将向量化后的数据进行存储
- **Retrieve** ：从大规模文本库中检索和查询相关的文本段落

**核心组件6：Callbacks**
Callbacks：回调机制，允许连接到 LLM 应用程序的各个阶段，可以监控和分析LangChain的运行情况，比如日志记录、监控、流传输等，以优化性能。

### 4.4 小结
- Model I/O模块：使用最多，也最简单
- Chains 模块： 最重要的模块
- Retrieval模块、Agents模块：大模型的主要落地场景

在这个基础上，其它组件要么是它们的辅助，要么只是完成常规应用程序的任务。

![alt text](/public/langchain/intro/20.png)

## 5. LangChain的helloworld

### 5.1 获取大模型

```python
#导入 dotenv 库的 load_dotenv 函数，用于加载环境变量文件（.env）中的配置
import dotenv
import os
from langchain_openai import ChatOpenAI

dotenv.load_dotenv() #加载当前目录下的 .env 文件
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
# 创建大模型实例
llm = ChatOpenAI(model="gpt-4o-mini") # 默认使用 gpt-3.5-turbo
# 直接提供问题，并调用llm
response = llm.invoke("什么是大模型？")
print(response)
```

其中，需要在当前工程下提供 .env 文件，文件中提供如下信息：

```
OPENAI_API_KEY="sk-cvUm8OddQbly.............AGgIHTm9kMH7Bf226G2" #你自己的密钥
OPENAI_BASE_URL="https://api.openai-proxy.org/v1" #url是固定值，统一写成这样
```
密钥来自于：https://www.closeai-asia.com/

### 5.2 使用提示词模板
我们也可以创建prompt template, 并引入一些变量到prompt template中，这样在应用的时候更加灵活。

```python
from langchain_core.prompts import ChatPromptTemplate
# 需要注意的一点是，这里需要指明具体的role，在这里是system和用户
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是世界级的技术文档编写者"),
    ("user", "{input}") # {input}为变量
])
# 我们可以把prompt和具体llm的调用和在一起。
chain = prompt | llm
message = chain.invoke({"input": "大模型中的LangChain是什么?"})
print(message)
# print(type(message))
```

### 5.3 使用输出解析器
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
# 初始化模型
llm = ChatOpenAI(model="gpt-4o-mini")
# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是世界级的技术文档编写者。"),
    ("user", "{input}")
])
# 使用输出解析器
# output_parser = StrOutputParser()
output_parser = JsonOutputParser()
# 将其添加到上一个链中
# chain = prompt | llm
chain = prompt | llm | output_parser
# 调用它并提出同样的问题。答案是一个字符串，而不是ChatMessage
# chain.invoke({"input": "LangChain是什么?"})
chain.invoke({"input": "LangChain是什么? 用JSON格式回复，问题用question，回答用answer"})
```

### 5.4 使用向量存储
使用一个简单的本地向量存储 FAISS，首先需要安装它

> pip install faiss-cpu
> 或者
> conda install faiss-cpu

> pip install langchain_community==0.3.7
> 或者
> conda install langchain_community==0.3.7

```python
# 导入和使用 WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
import bs4

loader = WebBaseLoader(
    web_path="https://www.gov.cn/xinwen/2020-06/01/content_5516649.htm",
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(id="UCAP-CONTENT"))
)
docs = loader.load()
# print(docs)
# 对于嵌入模型，这里通过 API调用

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 使用分割器分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50
)
documents = text_splitter.split_documents(docs)
print(len(documents))
# 向量存储 embeddings 会将 documents 中的每个文本片段转换为向量，并将这些向量存储在 FAISS向量数据库中
vector = FAISS.from_documents(documents, embeddings)
```

### 5.5 RAG(检索增强生成)
基于外部知识，增强大模型回复
```python
from langchain_core.prompts import PromptTemplate
retriever = vector.as_retriever()
retriever.search_kwargs = {"k": 3}
docs = retriever.invoke("建设用地使用权是什么？")
# for i,doc in enumerate(docs):
# print(f"⭐第{i+1}条规定：")
# print(doc)
# 6.定义提示词模版
prompt_template = """
你是一个问答机器人。
你的任务是根据下述给定的已知信息回答用户问题。
确保你的回
复完全依据下述已知信息。不要编造答案。
如果下述已知信息不足以回答用户的问题，请直接回
复"我
无法回答您的问题"。

已知信息:
{info}

用户问：
{question}

请用中文
回答用户问题。
"""
# 7.得到提示词模版对象
template = PromptTemplate.from_template(prompt_template)
# 8.得到提示词对象
prompt = template.format(info=docs, question='建设用地使用权是什么？')

## 9. 调用LLM
response = llm.invoke(prompt)
print(response.content)
```

### 5.6 使用Agent

```python
from langchain.tools.retriever import create_retriever_tool
# 检索器工具
retriever_tool = create_retriever_tool(
    retriever,
    "CivilCodeRetriever",
    "搜索有关中华人民共和国民法典的信息。关于中华人民共和国民法典的任何问题，您必须使用此工具!",
)
tools = [retriever_tool]
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

# https://smith.langchain.com/hub
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# 运行代理
agent_executor.invoke({"input": "建设用地使用权是什么"})
```