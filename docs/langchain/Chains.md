# LangChain Chains
## 1. Chains的基本使用
### 1.1 Chain的基本概念
Chain：链，用于将多个组件（提示模板、LLM模型、记忆、工具等）连接起来，形成可复用的 工作流 ，完成复杂的任务。
**Chain的核心思想**是通过组合不同的模块化单元，实现比单一组件更强大的功能。比如：
- 将 **`LLM`** 与 **`Prompt Template`**（提示模板）结合
- 将 **`LLM`** 与 **`输出解析器`** 结合
- 将 **`LLM`** 与 **`外部数据`** 结合，例如用于问答
- 将 **`LLM`** 与 **`长期记忆`** 结合，例如用于聊天历史记录
- 通过将 **`第一个LLM`** 的输出作为 **`第二个LLM`** 的输入，...，将多个LLM按顺序结合在一起
### 1.2 LCEL及其基本构成
使用LCEL，可以构造出结构最简单的Chain。
LangChain表达式语言（LCEL，LangChain Expression Language）是一种声明式方法，可以轻松地
将多个组件链接成 AI 工作流。它通过Python原生操作符（如管道符 | ）将组件连接成可执行流程，显著简化了AI应用的开发。

**LCEL的基本构成**：提示（Prompt）+ 模型（Model）+ 输出解析器（OutputParser）

```python
# 在这个链条中，用户输入被传递给提示模板，然后提示模板的输出被传递给模型，然后模型的输出被传递给输出解析器。
chain = prompt | model | output_parser
chain.invoke({"input":"What's your name?"})
```
- **Prompt**：Prompt 是一个 BasePromptTemplate，这意味着它接受一个模板变量的字典并生成一个`PromptValue`。PromptValue 可以传递给LLM（它以字符串作为输入）或 ChatModel（它以消息序列作为输入）。
- **Model**：将PromptValue传递给model。如果我们的 model是一个ChatModel，这意味着它
将输出一个`BaseMessage`。
- **OutputParser**：将model的输出传递给 output_parser，它是一个 BaseOutputParser，意味着它可以接受字符串或 BaseMessage 作为输入。
- **chain**：我们可以使用 | 运算符轻松创建这Chain。|运算符在LangChain中用于将两个元素组合在一起。
- **invoke**：所有LCEL对象都实现了 **`Runnable`** 协议，保证一致的调用方式（ **invoke / batch / stream** ）
> | 符号类似于shell⾥⾯管道操作符，它将不同的组件链接在⼀起，将前⼀个组件的输出作为下⼀个组件的输⼊，这就形成了⼀个 AI ⼯作流。

### 1.3 Runnable
Runnable是LangChain定义的一个抽象接口（Protocol），它`强制要求`所有LCEL组件实现一组标准方法：
```python
class Runnable(Protocol):
    def invoke(self, input: Any) -> Any: ... # 单输入单输出
    def batch(self, inputs: List[Any]) -> List[Any]: ... # 批量处理
    def stream(self, input: Any) -> Iterator[Any]: ... # 流式输出
    # 还有其他方法如 ainvoke（异步）等...
```
任何实现了这些方法的对象都被视为LCEL兼容组件。比如：聊天模型、提示词模板、输出解析器、检索器、代理(智能体)等。
每个 LCEL 对象都实现了 Runnable 接口，该接口定义了一组公共的调用方法。这使得 LCEL 对象链也自动支持这些调用成为可能。
**❓ 为什么需要统一调用方式？**
**传统问题**
假设没有统一协议：
- 提示词渲染用 `.format()`
- 模型调用用 `.generate()`
- 解析器解析用 `.parse()`
- 工具调用用 `.run()`
代码会变成：
```python
prompt_text = prompt.format(topic="猫") # 方法1
model_out = model.generate(prompt_text) # 方法2
result = parser.parse(model_out) # 方法3
```
**痛点：**
每个组件调用方式不同，组合时需要手动适配。

**LCEL解决方案**

通过 Runnable 协议统一：
```python
#（分步调用）
prompt_text = prompt.invoke({"topic": "猫"}) # 方法1
model_out = model.invoke(prompt_text) # 方法2
result = parser.invoke(model_out) # 方法3
#（LCEL管道式）
chain = prompt | model | parser # 用管道符组合
result = chain.invoke({"topic": "猫"}) # 所有组件统一用invoke
```
- **一致性**：无论组件的功能多复杂（模型/提示词/工具），调用方式完全相同
- **组合性**：管道操作符 | 背后自动处理类型匹配和中间结果传递

### 1.4 使用举例
***举例1***
```python
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
chat_model = ChatOpenAI(model="gpt-4o-mini")

prompt_template = PromptTemplate.from_template(template = "给我讲一个关于{topic}话题的简短笑话")
parser = StrOutputParser()
# 构建链式调用（LCEL语法）
chain = prompt_template | chat_model | parser
out_put = chain.invoke({"topic": "ice cream"})
print(out_put)
print(type(out_put))
```
```
为什么冰淇淋总是很快乐？
因为它知道⾃⼰是个“甜”⻆⾊！🍦😄
<class 'str'>
```
## 2. 传统Chain的使用
### 2.1 基础链：LLMChain
#### 2.1.1 使用说明
LCEL之前，最基础也最常见的链类型是LLMChain。
**这个链至少包括一个提示词模板（PromptTemplate），一个语言模型（LLM 或聊天模型）。**
::: warning
注意：LLMChain was deprecated in LangChain 0.1.17 and will be removed in 1.0 Use **Prompt | llm** instead
:::
**特点：**
- 用于`单次问答`，输入一个 Prompt，输出 LLM 的响应。
- 适合`无上下文`的简单任务（如翻译、摘要、分类等）。
- `无记忆`：无法自动维护聊天历史

#### 2.1.2 主要步骤
**1、配置任务链**：使用LLMChain类将任务与提示词结合，形成完整的任务链。
```python
chain = LLMChain(llm = llm, prompt = prompt_template)
```
**2、执行任务链**：使用invoke()等方法执行任务链，并获取生成结果。可以根据需要对输出进行处理和展示。
```python
result = chain.invoke(...)
print(result)
```
***举例：***
```python
# 1.导入相关包
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# 2.定义提示词模版对象
chat_template = ChatPromptTemplate.from_messages(
  [
    ("system","你是一位{area}领域具备丰富经验的高端技术人才"),
    ("human", "给我讲一个 {adjective} 笑话"),
  ]
)
# 3.定义模型
llm = ChatOpenAI(model="gpt-4o-mini")
# 4.定义LLMChain
llm_chain = LLMChain(llm=llm, prompt=chat_template, verbose=True)

# 5.调用LLMChain
response = llm_chain.invoke({"area":"互联网","adjective":"上班的"})
print(response)
```
> \> Entering new LLMChain chain...
Prompt after formatting:
System: 你是一位互联网领域具备丰富经验的高端技术人才
Human: 给我讲一个 上班的 笑话

> \> Finished chain.
{'area': '互联网', 'adjective': '上班的', 'text': '当然可以！这是一个上班的笑话：\n\n有一天，老板对员工说：“你知道为什么我总是把你的工作推迟吗？”\n\n员工好奇地问：“为什么呢？”\n\n老板微笑着回答：“因为我想让你们的工作保持新鲜感，每次都给你们一个新的截止日期，这样你们就能有更多的‘期待’！”\n\n员工无奈地说：“那我希望能把‘期待’换成薪水！”\n\n希望这个笑话能让你笑一笑！'}

### 2.2 顺序链之SimpleSequentialChain
顺序链（SequentialChain）允许将多个链顺序连接起来，每个Chain的输出作为下一个Chain的输入，
形成特定场景的流水线（Pipeline）。

**顺序链有两种类型：**
- 单个输入/输出：对应着 SimpleSequentialChain
- 多个输入/输出：对应着：SequentialChain
#### 2.2.1 说明
SimpleSequentialChain：最简单的顺序链，多个链`串联执行`，每个步骤都有`单一`的输入和输出，一个步骤的输出就是下一个步骤的输入，无需手动映射。
![alt text](/public/langchain/chain/1.png)
#### 2.2.2 使用举例
***举例:***
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
# 导入SimpleSequentialChain
from langchain.chains import SimpleSequentialChain
chainA_template = ChatPromptTemplate.from_messages(
  [
    ("system", "你是一位精通各领域知识的知名教授"),
    ("human", "请你尽可能详细的解释一下：{knowledge}"),
  ]
)
chainA_chains = LLMChain(llm=llm,prompt=chainA_template,verbose=True)
chainA_chains.invoke({"knowledge":"什么是LangChain？"})

chainB_template = ChatPromptTemplate.from_messages(
  [
    ("system", "你非常善于提取文本中的重要信息，并做出简短的总结"),
    ("human", "这是针对一个提问的完整的解释说明内容：{description}"),
    ("human", "请你根据上述说明，尽可能简短的输出重要的结论，请控制在20个字以"),
  ]
)
chainB_chains = LLMChain(llm=llm,prompt=chainB_template,verbose=True)
# 在chains参数中，按顺序传入LLMChain A 和LLMChain B
full_chain = SimpleSequentialChain(chains=[chainA_chains, chainB_chains], verbose=True)
full_chain.invoke({"input":"什么是langChain？"})
```
```
...
{'input': '什么是langChain？', 'output': 'LangChain是构建NLP应⽤的灵活框架，简化与语⾔模型的互动。'}
```
在这个过程中，因为`SimpleSequentialChain`定义的是顺序链，所以在`chains`参数中传递的列表要按照顺序来进行传入，即LLMChain A 要在LLMChain B之前。同时，在调用时，不再使用LLMChain A中定义的`{knowledge}`参数，也不是LLMChainB中定义的`{description}`参数，而是要使用`input`进行变量的传递。
```python
class SimpleSequentialChain(Chain):
    """Simple chain where the outputs of one step feed directly into next."""
    chains: List[Chain]
    strip_outputs: bool = False
    input_key: str = "input" #: :meta private:
    output_key: str = "output" #: :meta private:
```
### 2.3 顺序链之 SequentialChain
#### 2.3.1 说明
SequentialChain：更通用的顺序链，具体来说：
- **`多变量支持`**：允许不同子链有独立的输入/输出变量。
- **`灵活映射`**：需 显式定义 变量如何从一个链传递到下一个链。即精准地命名输入关键字和输出关键字，来明确链之间的关系。
- **`复杂流程控制`**：支持分支、条件逻辑（分别通过 input_variables 和 output_variables 配置输入和输出）。
![alt text](/public/langchain/chain/2.png)
#### 2.3.2 使用举例
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import SequentialChain
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from openai import OpenAI
import os
# 创建大模型实例
llm = ChatOpenAI(model="gpt-4o-mini")
schainA_template = ChatPromptTemplate.from_messages(
  [
    ("system", "你是一位精通各领域知识的知名教授"),
    ("human", "请你先尽可能详细的解释一下：{knowledge}，并且{action}")
  ]
)
schainA_chains = LLMChain(llm=llm,prompt=schainA_template,verbose=True,output_key="schainA_chains_key")
# schainA_chains.invoke({
# "knowledge": "中国的篮球怎么样？",
# "action": "举一个实际的例子"
# }
# )
schainB_template = ChatPromptTemplate.from_messages(
  [
    ("system", "你非常善于提取文本中的重要信息，并做出简短的总结"),
    ("human", "这是针对一个提问完整的解释说明内容：{schainA_chains_key}"),
    ("human", "请你根据上述说明，尽可能简短的输出重要的结论，请控制在100个字以"),
  ]
)
schainB_chains = LLMChain(llm=llm,prompt=schainB_template,verbose=True,output_key='schainB_chains_key')

Seq_chain = SequentialChain(
  chains=[schainA_chains, schainB_chains],
  input_variables=["knowledge", "action"],
  output_variables=["schainA_chains_key","schainB_chains_key"],
  verbose=True
)
response = Seq_chain.invoke(
  {
    "knowledge":"中国足球为什么踢得烂",
    "action":"举一个实际的例子"
  }
)
print(response)
```
还可以单独输出：
```python
print(response["schainA_chains_key"])
print(response["schainB_chains_key"])
```
#### 2.3.3 顺序链使用场景
**场景**：多数据源处理

举例：根据产品名
> 1. 查询数据库获取价格
> 2. 生成促销文案

**使用 SimpleSequentialChain（会失败）**
```
# 假设链1返回 {"price": 100}, 链2需要 {product: "xx", price: xx}
# 结构不匹配，无法自动传递！
```

**使用 SequentialChain（正确方式）**
```python
from langchain.chains import SequentialChain
# 创建大模型实例
llm = ChatOpenAI(model="gpt-4o-mini")
# 第1环节：
query_chain = LLMChain(
  llm=llm,
  prompt=PromptTemplate.from_template(template="请模拟查询{product}的市场价格，直接返回一个合理的价格数字（如6999），不要包含任何其他文字或代码"),verbose=True,
  output_key="price")
# 第2环节：
promo_chain = LLMChain(
  llm=llm,
  prompt=PromptTemplate.from_template(template="为{product}（售价：{price}元）创作一篇50字以内的促销文案，要求突出产品卖点"),verbose=True,output_key="promo_text"
)
sequential_chain = SequentialChain(
  chains=[query_chain, promo_chain],
  verbose=True,input_variables=["product"], # 初始输入
  output_variables=["price", "promo_text"], # 输出价格和文案
)
result = sequential_chain.invoke({"product": "iPhone16"})
print(result)
# print(result["price"]
# print(result["promo_text"]
```
```
{
  'product': 'iPhone16',
  'price': '6999',
  'promo_text': '全新iPhone 16，6999元，体验超⾼清影像与强劲性能，A17芯⽚助你畅享流畅操作。⽆与伦⽐的续航与创新设计，期待你的每⼀次发现，开启未来智能⽣活！尽快抢购，名额有限！'
}
```

### 2.4 数学链LLMMathChain(了解)
LLMMathChain将用户问题转换为数学问题，然后将数学问题转换为可以使用**Python**的**numexpr**库执行的表达式。使用运行此代码的输出来回答问题。

使用LLMMathChain，需要安装numexpr库
> pip install numexpr
```python
from langchain.chains import LLMMathChain
import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
# 创建大模型实例
llm = ChatOpenAI(model="gpt-4o-mini")
# 创建链
llm_math = LLMMathChain.from_llm(llm)
# 执行链
res = llm_math.invoke("10 ** 3 + 100的结果是多少？")
print(res)
```
```
{'question': '10 ** 3 + 100的结果是多少？', 'answer': 'Answer: 1100'}
```
### 2.5 路由链 RouterChain (了解)
路由链（RouterChain）用于创建可以 动态选择下一条链 的链。可以自动分析用户的需求，然后引导到最适合的链中执行，获取响应并返回最终结果。

比如，我们目前有三类chain，分别对应三种学科的问题解答。我们的输入内容也是与这三种学科对应，但是随机的，比如第一次输入数学问题、第二次有可能是历史问题... 这时候期待的效果是：可以根据输入的内容是什么，自动将其应用到对应的子链中。RouterChain就为我们提供了这样一种能力。
![alt text](/public/langchain/chain/3.png)
:::info 
它会⾸先决定将要传递下去的⼦链，然后把输⼊传递给那个链。并且在设置的时候需要注意为其**设置默认chain**，以兼容输⼊内容不满⾜任意⼀项时的情况。
:::
**RouterChain图示：**
![alt text](/public/langchain/chain/4.png)
### 2.6 文档链 StuffDocumentsChain(了解)
StuffDocumentsChain 是一种文档处理链，它的核心作用是将`多个文档内容合并`（“填充”或“塞入”）到单个提示（prompt）中，然后传递给语言模型（LLM）进行处理。

**使用场景**：由于所有文档被完整拼接，LLM能同时看到全部内容，所以适合需要全局理解的任务，如总结、问答、对比分析等。但注意，仅适合处理`少量/中等长度文档`的场景。

***举例：***
```python
#1.导入相关包
from langchain.chains import StuffDocumentsChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
# 2.加载PDF
loader = PyPDFLoader("./asset/example/loader.pdf")
#3.定义提示词
prompt_template = """对以下文字做简洁的总结:
{text}
简洁的总结:"""
# 4.定义提示词模版
prompt = PromptTemplate.from_template(prompt_template)
# 5.定义模型
llm = ChatOpenAI(model="gpt-4o-mini")
# 6.定义LLM链
llm_chain = LLMChain(llm=llm, prompt=prompt)
# 7.定义文档链
stuff_chain = StuffDocumentsChain(
  llm_chain=llm_chain,
  document_variable_name="text", # 在 prompt 模板中，文档内容应该用哪个变量名表示
) #document_variable_name="text" 告诉 StuffDocumentsChain 把合并后的文档内容填充到 {text}变量中"。

# 8.加载pdf文档
docs = loader.load()
# 9.执行链
res=stuff_chain.invoke(docs)
#print(res)
print(res["output_text"])
```
```
蒂法·洛克哈特是电子游戏《最终幻想VII》及其相关作品中的虚构角色，由野村哲也设计。她是主角克劳德的青梅竹马，拥有强大的格斗技能，并在游戏中扮演重要角色。蒂法在多个游戏和媒体中客串登场，并被认为是电子游戏中坚强、独立的女性角色代表。她的形象和性格受到广泛赞誉，成为了电子游戏界的标志性人物之一。
```
## 3. 基于LCEL构建的Chains的类型
前面讲解的都是Legacy Chains，下面看最新的基于LCEL构建的Chains。
```
create_sql_query_chain
create_stuff_documents_chain
create_openai_fn_runnable
create_structured_output_runnable
load_query_constructor_runnable
create_history_aware_retriever
create_retrieval_chain
```
### 3.1 create_sql_query_chain
create_sql_query_chain，SQL查询链，是创建生成SQL查询的链，用于将`自然语言`转换成`数据库的SQL查询`。

***举例1：***
这里使用MySQL数据库，需要安装pymysql
> pip install pymysql
```python
from langchain_openai import ChatOpenAI
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
# 连接 MySQL 数据库
db_user = "root"
db_password = "abc123" #根据自己的密码填写
db_host = "127.0.0.1"
db_port = "3306"
db_name = "atguigudb"
# 固定格式：mysql+pymysql://用户名:密码@ip地址:端口号/数据库名
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

print("哪种数据库：", db.dialect)
print("获取数据表：", db.get_usable_table_names())
# 执行查询
res = db.run("SELECT count(*) FROM employees;")
print("查询结果：", res)

# 创建大模型实例
llm = ChatOpenAI(model="gpt-4o-mini")
# 调用Chain
chain = create_sql_query_chain(llm=llm, db=db)
# response = chain.invoke({"question": "数据表employees中哪个员工工资高？"})
# print(response)
# response = chain.invoke({"question": "查询departments表中一共有多少个部门？"})
# print(response)
# response = chain.invoke({"question": "查询last_name叫King的基本情况"})
# print(response)
# # 限制使用的表
response = chain.invoke({"question": "一共有多少个员工？","table_names_to_use":["employees"]})
print(response)
```
### 3.2 create_stuff_documents_chain(了解)
create_stuff_documents_chain用于将`多个文档内容`合并成`单个长文本`的链式工具，并一次性传递给LLM处理（而不是分多次处理）。
适合场景：
- 保持上下文完整，适合需要全局理解所有文档内容的任务（如总结、问答）
- 适合处理`少量/中等长度文档`的场景。
***举例：***多文档摘要

```python
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
# 定义提示词模板
prompt = PromptTemplate.from_template("""如下文档{docs}中说，香蕉是什么颜色的？""")
# 创建链
llm = ChatOpenAI(model="gpt-4o-mini")
chain = create_stuff_documents_chain(llm, prompt, document_variable_name="docs")
# 文档输入
docs = [
  Document(page_content="苹果，学名Malus pumila Mill.，别称西洋苹果、柰，属于蔷薇科苹果属的植物。苹果是全球最广泛种植和销售的水果之一，具有悠久的栽培历史和广泛的分布范围。苹果的原始种群主要起源于中亚的天山山脉附近，尤其是现代哈萨克斯坦的阿拉木图地区，提供了所有现代苹果品种的基因库。苹果通过早期的贸易路线，如丝绸之路，从中亚向外扩散到全球各地。"),
  Document(page_content="香蕉是白色的水果，主要产自热带地区。"),
  Document(page_content="蓝莓是蓝色的浆果，含有抗氧化物质。")
]
# 执行摘要
chain.invoke({"docs": docs})
```
```
'⾹蕉是⻩⾊的⽔果，通常在成熟时呈现明亮的⻩⾊。你提到的描述“⽩⾊的⽔果”可能是对⾹蕉未成熟状态的误解。在成熟阶段，它们⼤多数情况下是⻩⾊的。'
```