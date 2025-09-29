# LangChain Model I/O

## 1. Model I/O介绍

Model I/O 模块是与语言模型（LLMs）进行交互的`核心组件`，在整个框架中有着很重要的地位。所谓的Model I/O，包括输入提示(Format)、调用模型(Predict)、输出解析(Parse)。分别对应着`Prompt Template`，`Model`和`Output Parser`。

## 2. Model I/O之调用模型1
LangChain作为一个“工具”，不提供任何LLMs，而是依赖于第三方集成各种大模型。比如，将OpenAI、Anthropic、HuggingFace、LlaMA、阿里Qwen、ChatGLM等平台的模型无缝接入到你的应用。

### 2.1 模型的不同分类方式

**角度1：按照模型功能的不同：**
  - 非对话模型（LLMs、Text Model）
  - 对话模型（Chat Models）（`推荐`）
  - 嵌入模型（Embedding Models）(暂不考虑)

**角度2：模型调用时，几个重要参数的书写位置的不同：**
  - 硬编码：写在代码文件中
  - 使用环境变量
  - 使用配置文件（`推荐`）

**角度3：具体调用的API**
  - OpenAI提供的API
  - 其它大模型自家提供的API
  - LangChain的统一方式调用API（`推荐`）
  
::: info
`背景小知识`：OpenAI的GPT系列模型影响了⼤模型技术发展的开发范式和标准。所以⽆论是Qwen、ChatGLM等模型，它们的使⽤⽅法和函数调⽤逻辑基本 遵循OpenAI定义的规范 ，没有太⼤差异。这就使得⼤部分的开源项⽬能够通过⼀个较为通⽤的接口来接⼊和使⽤不同的模型。
:::

### 2.2 角度1出发：按照功能不同举例
**类型1：LLMs(非对话模型)**
LLMs，也叫Text Model、非对话模型，是许多语言模型应用程序的支柱。主要特点如下：
  - **输入**：接受`文本字符串`或`PromptValue`对象
  - **输出**：总是返回`文本字符串`
  - **适用场景**：仅需单次文本生成任务（如摘要生成、翻译、代码生成、单次问答）或对接不支持消息
  - **结构的旧模型**（如部分本地部署模型）（`言外之意，优先推荐ChatModel`）不支持多轮对话上下文。每次调用独立处理输入，无法自动关联历史对话（需手动拼接历史文本）。
  - **局限性**：无法处理角色分工或复杂对话逻辑。

```python
import os
import dotenv
from langchain_openai import OpenAI

# 加载配置文件
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_BASE_URL'] = os.getenv('OPENAI_BASE_URL')

# 创建模型
model = OpenAI(model='gpt-4o-mini')
# 模型调用
response = model.invoke("帮我写一首春天的诗歌")
# 打印结果
print(response)
```
```
春天来了，大地复苏，
万物复苏，精神抖擞。
花儿绽放，鸟儿啁啾，
一切都充满生机与活力。
```

**类型2：Chat Models(对话模型)**

ChatModels，也叫聊天模型、对话模型，底层使用LLMs。

**大语言模型调用，以`ChatModel`为主！**

主要特点如下：

**输入**：接收消息列表 `List[BaseMessage]` 或 `PromptValue` ，每条消息需指定角色（如SystemMessage、HumanMessage、AIMessage）。

**输出**：总是返回带角色的`消息对象`(`BaseMessage子类`），通常是`AIMessage`。

**原生支持多轮对话**：通过消息列表维护上下文（例如：[`SystemMessage`,`HumanMessage`,`AIMessage`,...]），模型可基于完整对话历史生成回复。

**适用场景**：对话系统（如客服机器人、长期交互的AI助手）。

***举例***
```python
import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage
# 加载配置文件
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_BASE_URL'] = os.getenv('OPENAI_BASE_URL')

# 创建模型文件
chat_model = ChatOpenAI(model="gpt-4o-mini")

# 创建message
messages = [
    SystemMessage(content="我是人工智能助手小智"),
    HumanMessage(content="你好，我是徐赞，很高兴认识你")
]
# 输入消息列表
response = chat_model.invoke(messages)
# 打印输出内容和结果类型
print(type(response))
print(response.content)
```
```
<class 'langchain_core.messages.ai.AIMessage'>
你好，徐赞！很高兴认识你！请问有什么我可以帮助你的吗？
```
**类型3：Embedding Model(嵌入模型)**
**Embedding Model**：也叫文本嵌入模型，这些模型将`文本`作为输入并返回浮点数列表，也就是Embedding。

```python
import os
import dotenv
from langchain_openai import OpenAIEmbeddings

# 加载配置文件
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_BASE_URL'] = os.getenv('OPENAI_BASE_URL')

# 创建模型文件
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# 调用嵌入模型进行文档的向量化
response = embedding_model.embed_query("我是文档中的数据")
# 打印结果
print(response)
```
```
[-0.032839685678482056, 0.002757745562121272, -0.010489282198250294, ...]
```
### 2.3 角度2出发：参数位置不同举例
这里以`LangChain`的API为准，使用的对话模型，进行测试。
#### 2.3.1 模型的调用主要方法及参数
**相关方法及属性**
  - `OpenAI(...)/ChatOpenAI(...)`：创建一个模型对象（非对话类/对话类）
  - `model.invoke(xxx)`：执行调用，将用户输入发送给模型
  - `.content`：提取模型返回的实际文本内容
模型调用函数使用时需初始化模型，并设置必要的参数。

**1）必须设置的参数：**
  - `base_url`：大模型`API`服务的根地址
  - `api_key` ：用于身份验证的密钥，由大模型服务商（如OpenAI、百度千帆）提供
  - `model/model_name`：指定要调用的具体大模型名称（如 `gpt-4-turbo`、`ERNIE-3.5-8K`等）

**2）其它参数：**
- `temperature`：温度，控制生成文本的“随机性”，取值范围为0～1。
  - `值越低` → 输出越确定、保守（适合事实回答）
  - `值越高` → 输出越多样、有创意（适合创意写作）
  通常，根据需要设置如下：
  - 精确模式（0.5或更低）：生成的文本更加安全可靠，但可能缺乏创意和多样性。
  - 平衡模式（通常是0.8）：生成的文本通常既有一定的多样性，又能保持较好的连贯性和准确
性。

  - 创意模式（通常是1）：生成的文本更有创意，但也更容易出现语法错误或不合逻辑的内容。
- max_tokens ：限制生成文本的最大长度，防止输出过长。

#### Token是什么？
**`基本单位`**: 大模型处理文本的最小单位是token（相当于自然语言中的词或字），输出时逐个token依次生成。

**`收费依据`**：大语言模型(LLM)通常也是以token的数量作为其计量(或收费)的依据。
- 1个Token≈1-1.8个汉字，1个Token≈3-4个英文字母
- Token与字符转化的可视化工具：
  - OpenAI提供：https://platform.openai.com/tokenizer
  - 百度智能云提供：https://console.bce.baidu.com/support/#/tokenizer

**max_tokens设置建议：**
- 客服短回复：128-256。比如：生成一句客服回复（如“订单已发货，预计明天送达”）
- 常规对话、多轮对话：512-1024
- 长内容生成：1024-4096。比如：生成一篇产品说明书（包含功能、使用方法等结构）

#### 2.3.2 模型调用推荐平台：closeai
这里推荐使用的平台：
考虑到OpenAI等模型在国内访问及充值的不便，大家可以使用CloseAI网站注册和充值，具体费用自理。
https://www.closeai-asia.com

#### 2.3.3 方式1：硬编码
直接将 API Key和模型参数写入代码，仅适用于临时测试，存在密钥泄露风险，在`生产环境不推荐`。

```python
# 硬编码 API Key 和模型参数
llm = ChatOpenAI(
  api_key="sk-xxxxxxxxx", # 明文暴露密钥
  base_url="https://api.openai-proxy.org/v1",
  model="gpt-3.5-turbo",
)
```
#### 2.3.4 方式2：配置环境变量
***方式1：终端设置环境变量（临时生效）：***
```js
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx" # Linux/Mac
set OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx" # Windows
```
***方式2：从PyCharm设置***
![alt text](/public/langchain/modelIO/1.png)
![alt text](/public/langchain/modelIO/2.png)

#### 2.3.5 方式3：使用.env配置文件
使用 python-dotenv 加载本地配置文件，支持多环境管理（开发/生产）。

**1）安装依赖**
```python
pip install python-dotenv
#或者
conda install python-dotenv
```
**2）创建 .env 文件（项目根目录）：**
```python
OPENAI_API_KEY="sk-xxxxxxxxx" # 需填写自己的API KEY
OPENAI_BASE_URL="https://api.openai-proxy.org/v1"
```
***方式1:***
```python
## 核心代码 ##
load_dotenv() # 自动加载 .env 文件
llm = ChatOpenAI(
  api_key=os.getenv("OPENAI_API_KEY"), # 安全读取
  base_url=os.getenv("OPENAI_BASE_URL"),
  model="gpt-4o-mini",
)
```

***方式2：给os内部的环境变量赋值***
```python
## 核心代码 ##
load_dotenv() # 自动加载 .env 文件
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")
chat_model = ChatOpenAI(
  model="gpt-4o-mini",
  temperature=0.7,
  max_tokens=300,
)
```
**小结：**

|     方式     | 安全性 |  持久性  |      适用场景      |
| :----------: | :----: | :------: | :----------------: |
|    硬编码    |  ⚠ 低  |  ❌ 临时  |    本地快速测试    |
|   环境变量   |  ✅ 中  | ⚠ 会话级 |    短期开发调试    |
| .env配置文件 | ✅✅ 高  |  ✅ 永久  | 生产环境、团队协作 |

## 3. Model I/O之调用模型2
### 3.1 关于对话模型的Message(消息)
聊天模型，出了将字符串作为输入外，还可以使用`聊天消息`作为输入，并返回`聊天消息`作为输出。

**LangChain有一些内置的消息类型**：
- 🔥**SystemMessage**：设定AI行为规则或背景信息。比如设定AI的初始状态、行为模式或对话的总体目标。比如“作为一个代码专家”，或者“返回json格式”。通常作为输入消息序列中的第一个传递。
- 🔥**HumanMessage**：表示来自用户输入。比如“实现一个快速排序方法”
- 🔥**AIMessage**：存储AI回复的内容。这可以是文本，也可以是调用工具的请求
- **ChatMessage** ：可以自定义角色的通用消息类型
- **FunctionMessage/ToolMessage** ：函数调用/工具消息，用于函数调用结果的消息类型

::: warning
FunctionMessage和ToolMessage分别是在函数调⽤和⼯具调⽤场景下才会使⽤的特殊消息类型，HumanMessage、AIMessage和SystemMessage才是最常⽤的消息类型。
:::

***举例：***
```python
import os
from langchain_core.messages import SystemMessage,HumanMessage
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
chat_model = ChatOpenAI(
  model="gpt-4o-mini",
)
# 组成消息列表
messages = [
  SystemMessage(content="你是一个擅长人工智能相关学科的专家"),
  HumanMessage(content="请解释一下什么是机器学习？")
]
response = chat_model.invoke(messages)
print(response.content)
print(type(response)) #<class 'langchain_core.messages.ai.AIMessage'>
```

### 3.2 关于多轮对话与上下文记忆
***举例：***
大模型本身就无记忆功能，在当次对话中才能有一定记忆功能。
```python
import os
import dotenv
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
chat_model = ChatOpenAI(
  model="gpt-4o-mini"
)
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage
messages = [
  SystemMessage(content="我是一个人工智能助手，我的名字叫小智"),
  HumanMessage(content="人工智能英文怎么说？"),
  AIMessage(content="AI"),
  HumanMessage(content="你叫什么名字"),
]
messages1 = [
  SystemMessage(content="我是一个人工智能助手，我的名字叫小智"),
  HumanMessage(content="很
  高兴认识你"),
  AIMessage(content="我也很
  高兴认识你"),
  HumanMessage(content="你叫什么名字"),
]
messages2 = [
  SystemMessage(content="我是一个人工智能助手，我的名字叫小智"),
  HumanMessage(content="人工智能英文怎么说？"),
  AIMessage(content="AI"),
  HumanMessage(content="你叫什么名字"),
]
chat_model.invoke(messages2)
```
### 3.3 关于模型调用的方法
为了尽可能简化自定义链的创建，我们实现了一个"`Runnable`"协议。许多LangChain组件实现了`Runnable`协议，包括聊天模型、提示词模板、输出解析器、检索器、代理(智能体)等。

**Runnable 定义的公共的调用方法如下：**
  - **invoke**: 处理单条输入，等待LLM完全推理完成后再返回调用结果
  - **stream**: 流式响应，逐字输出LLM的响应结果
  - **batch**: 处理批量输入
  这些也有相应的异步方法，应该与 `asyncio` 的 `await` 语法一起使用以实现并发：
  - **astream** : 异步流式响应
  - **ainvoke** : 异步处理单条输入
  - **abatch** : 异步处理批量输入
  - **astream_log** : 异步流式返回中间步骤，以及最终响应
  - **astream_events** : （测试版）异步流式返回链中发生的事件（`langchain-core`0.1.14中引入）
#### 3.3.1 流式输出与非流式输出
在Langchain中，语言模型的输出分为了两种主要的模式：`流式输出`与`非流式输出`。
- 流式输出：一种更具交互感 的模型输出方式，用户不再需要等待完整答案，而是能看到模型 逐个**token**地实时返回内容。
- 非流式输出：这是Langchain与LLM交互时的**默认行为**，是最简单、最稳定的语言模型调用方式。当用户发出请求后，系统在后台等待模型**生成完整响应**，然后一次性将全部结果返回。

**非流式输出：**
```python
import os
import dotenv
from langchain
from langchain_core.messages import HumanMessage_openai import ChatOpenAI
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
#初始化大模型
chat_model = ChatOpenAI(model="gpt-4o-mini")
# 创建消息
messages = [HumanMessage(content="你好，请介绍一下自己")]
# 非流式调用LLM获取响应
response = chat_model.invoke(messages)
# 打印响应内容
print(response)
```
输出结果如下，是直接全部输出的。

```json
content='你好！我是一个人工智能助手，旨在回答问题、提供信息和帮助解决各种问题。....
```
**流式输出：**
一种更具交互感的模型输出方式，用户不再需要等待完整答案，而是能看到模型逐个 token 地实时返回内容。适合构建强调“实时反馈”的应用，如聊天机器人、写作助手等。
Langchain中通过设置`streaming=True`并配合`回调机制`来启用流式输出。

```python
import os
import dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
# 初始化大模型
chat_model = ChatOpenAI(model="gpt-4o-mini",
  streaming=True # 启用流式输出
)
# 创建消息
messages = [HumanMessage(content="你好，请介绍一下自己")]
# 流式调用LLM获取响应
print("开始流式输出：")
for chunk in chat_model.stream(messages):
  # 逐个打印内容块
  # 刷新缓冲区 (无换行符，缓冲区未刷新，内容可能不会立即显示)
  print(chunk.content, end="", flush=True) 
print("\n流式输出结束")
```
#### 3.3.2 批量调用
```python
import os
import dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
# 初始化大模型
chat_model = ChatOpenAI(model="gpt-4o-mini")
messages1 = [SystemMessage(content="你是一位乐于助人的智能小助手"),
HumanMessage(content="请帮我介绍一下什么是机器学习"), ]
messages2 = [SystemMessage(content="你是一位乐于助人的智能小助手"),
HumanMessage(content="请帮我介绍一下什么是AIGC"), ]
messages3 = [SystemMessage(content="你是一位乐于助人的智能小助手"),
HumanMessage(content="请帮我介绍一下什么是大模型技术"), ]
messages = [messages1, messages2, messages3]
# 调用batch
response = chat_model.batch(messages)
print(response)
```
#### 3.3.3 同步调用与异步调用(了解)
**同步调用**
```python
import time
def call_model():
  # 模拟同步API调用
  print("开始调用模型...")
  time.sleep(5) # 模拟调用等待,单位：秒
  print("模型调用完成。")
def perform_other_tasks():
  # 模拟执行其他任务
  for i in range(5):
    print(f"执行其他任务 {i + 1}")
    time.sleep(1) # 单位：秒
def main():
  start_time = time.time()
  call_model()
  perform_other_tasks()
  end_time = time.time()
  total_time = end_time - start_time
  return f"总共耗时：{total_time}秒"
# 运行同步任务并打印完成时间
main_time = main()
print(main_time)
```
```
开始调⽤模型
...
模型调⽤完成。
执⾏其他任务 1
执⾏其他任务 2
执⾏其他任务 3
执⾏其他任务 4
执⾏其他任务 5
总共耗时：10.061029434204102秒
```
> 之前的`llm.invoke(...)`本质上是一个同步调用。每个操作依次执行，直到当前操作完成后才开始下一个操作，从而导致总的执行时间是各个操作时间的总和。

**异步调用**
异步调用，允许程序在等待某些操作完成时继续执行其他任务，而不是阻塞等待。这在处理I/O操作（如网络请求、文件读写等）时特别有用，可以显著提高程序的效率和响应性。

***举例：***
```python
import asyncio
import time
async def async_call(llm):
  await asyncio.sleep(5) # 模拟异步操作
  print("异步调用完成")
async def perform_other_tasks():
  await asyncio.sleep(5) # 模拟异步操作
  print("其他任务完成")
async def run_async_tasks():
  start_time = time.time()
  await asyncio.gather(
    async_call(None), # 示例调用，替换None为模拟的LLM对象
    perform_other_tasks()
  )
  end_time = time.time()
  return f"总共耗时：{end_time - start_time}秒"
# 正确运行异步任务的方式
if __name__ == "__main__":
  # 使用 asyncio.run() 来启动异步程序
  result = asyncio.run(run_async_tasks())
  print(result)
```
```
异步调⽤完成
其他任务完成
总共耗时：5.001038551330566秒
```
> 使用`asyncio.gather()`并行执行时，理想情况下，因为两个任务几乎同时开始，它们的执行时间将重叠。如果两个任务的执行时间相同（这里都是5秒），那么总执行时间应该接近单个任务的执行时间，而不是两者时间之和。

## 4. Model I/O之Prompt Template
Prompt Template，通过模板管理大模型的输入。
### 4.1 介绍与分类
Prompt Template 是LangChain中的一个概念，接收用户输入，返回一个传递给LLM的信息（即提示词prompt）。
在应用开发中，固定的提示词限制了模型的灵活性和适用范围。所以，prompt template 是一个`模板化的字符串`，你可以将`变量插入到模板`中，从而创建出不同的提示。调用时：
  - 以`字典`作为输入，其中每个键代表要填充的提示模板中的变量。
  - 输出一个 PromptValue 。这个`PromptValue`可以传递给 LLM 或 ChatModel，并且还可以转换为字符串或消息列表。
**有几种不同类型的提示模板：**
  - PromptTemplate ：LLM提示模板，用于生成字符串提示。它使用 Python 的字符串来模板提示。
  - ChatPromptTemplate ：聊天提示模板，用于组合各种角色的消息模板，传入聊天模型。
  - XxxMessagePromptTemplate ：消息模板词模板，包括：SystemMessagePromptTemplate、HumanMessagePromptTemplate、AIMessagePromptTemplate、ChatMessagePromptTemplate等
  - FewShotPromptTemplate ：样本提示词模板，通过示例来教模型如何回答
  - PipelinePrompt ：管道提示词模板，用于把几个提示词组合在一起使用。
  - 自定义模板 ：允许基于其它模板类来定制自己的提示词模板。
### 4.2 具体使用：PromptTemplate
#### 4.2.1 使用说明
PromptTemplate类，用于快速构建`包含变量`的提示词模板，并通过`传入不同的参数值`生成自定义的提示词。
**主要参数介绍：**
- **template**：定义提示词模板的字符串，其中包含`文本`和`变量占位符（如{name}）`；
- **input_variables**：列表，指定了模板中使用的变量名称，在调用模板时被替换；
- **partial_variables**：字典，用于定义模板中一些固定的变量名。这些值不需要再每次调用时被替换。
**函数介绍：**
- **format()**：给input_variables变量赋值，并返回提示词。利用format() 进行格式化时就一定要赋值，否则会报错。当在template中未设置input_variables，则会自动忽略。
#### 4.2.2 两种实例化方式
**方式1：使用构造方法**

***举例：***
```python
from langchain.prompts import PromptTemplate
# 定义模板：描述主题的应用
template = PromptTemplate(template="请简要描述{topic}的应用。",input_variables=["topic"])
print(template)
# 使用模板生成提示词
prompt_1 = template.format(topic="机器学习")
prompt_2 = template.format(topic="自然语言处理")
print("提示词1:", prompt_1)
print("提示词2:", prompt_2)
```
```
input_variables=['topic'] input_types={} partial_variables={} template='请简要描述{topic}的应⽤。'
提⽰词1: 请简要描述机器学习的应⽤。
提⽰词2: 请简要描述⾃然语⾔处理的应⽤。
```
**方式2：调用from_template()模板支持任意数量的变量，包括不含变量：**

***举例：***
```python
from langchain.prompts import PromptTemplate
prompt_template = PromptTemplate.from_template("请给我一个关于{topic}的{type}解释。")
#传入模板中的变量名
prompt = prompt_template.format(type="详细", topic="量子力学")
print(prompt)
# 输出：请给我⼀个关于量⼦⼒学的详细解释。
```

#### 4.2.3 两种新的结构形式
**形式1:部分提示词模板**
在生成prompt前就已经提前初始化部分的提示词，实际进一步导入模版的时候只导入除已初始化的变量即可。

***举例1:***

- **方式1：实例化过程中使用partial_variables变量**
```python
from langchain.prompts import PromptTemplate
#方式1：
template2 = PromptTemplate(
  template="{foo}{bar}",
  input_variables=["foo","bar"],
  partial_variables={"foo": "hello"}
)
prompt2 = template2.format(bar="world")
print(prompt2)
```
- **方式2：使用 PromptTemplate.partial() 方法创建部分提示模板**
```python
from langchain.prompts import PromptTemplate
template1 = PromptTemplate(
  template="{foo}{bar}",
  input_variables=["foo", "bar"])
#方式2：partial()调用完之后，不会对调用者这个模板对象产生影响
partial_template1 = template1.partial(foo="hello")
prompt1 = partial_template1.format(bar="world")
print(prompt1)
```
***举例2:***
```python
from langchain_core.prompts import PromptTemplate
# 完整模板
full_template = """你是一个{role}，请用{style}风格回答：
问题：{question}
答案："""
# 预填充角色和风格
partial_template = PromptTemplate.from_template(full_template).partial(role="资深厨师",style="专业但幽默")
# 只需提供剩余变量
print(partial_template.format(question="如何煎牛排？"))
```
```
你是⼀个资深厨师，请⽤专业但幽默⻛格回答：
问题：如何煎⽜排？
答案：
```
***举例3:***
```python
prompt_template = PromptTemplate.from_template(template = "请评价{product}的优缺点，包括{aspect1}和{aspect2}。",
partial_variables= {"aspect1":"电池","aspect2":"屏幕"})
prompt= prompt_template.format(product="笔记本电脑")
print(prompt)
```
**形式2：组合提示词(了解)**

***举例：***
```python
from langchain_core.prompts import PromptTemplate
template = (PromptTemplate.from_template("Tell me a joke about {topic}")
  + ", make it funny"
  + "\n\nand in {language}")
prompt = template.format(topic="sports", language="spanish")
print(prompt)
```

```
Tell me a joke about sports,makes it funny
and in spanish
```
#### 4.2.4 format() 与 invoke()
只要对象是RunnableSerializable接口类型，都可以使用invoke()，替换前面使用format()的调用方式。
format()，返回值为字符串类型；invoke()，返回值为PromptValue类型，接着调用to_string()返回字符串。

***举例1：***
```python
#1.导入相关的包
from langchain_core.prompts import PromptTemplate
# 2.定义提示词模版对象
prompt_template = PromptTemplate.from_template("Tell me a {adjective} joke about {content}.")
# 3.默认使用f-string进行格式化（返回格式好的字符串）
prompt_template.invoke({"adjective":"funny", "content":"chickens"})
```
***举例2：***
```python
#1.导入相关的包
from langchain_core.prompts import PromptTemplate
# 2.使用初始化器进行实例化
prompt = PromptTemplate(input_variables=["adjective", "content"],
template="Tell me a {adjective} joke about {content}")
# 3. PromptTemplate底层是RunnableSerializable接口 所以可以直接使用invoke()调用
prompt.invoke({"adjective": "funny", "content": "chickens"})
```
***举例3：***
```python
from langchain_core.prompts import PromptTemplate
prompt_template = (PromptTemplate.from_template("Tell me a joke about {topic}")+ ", make it funny"+ " and in {language}")
prompt = prompt_template.invoke({"topic":"sports","language":"spanish"})
print(prompt)
```
#### 4.2.5 结合LLM调用
以对话大模型举例：
```python
import os
import dotenv
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
llm = ChatOpenAI(model="gpt-4o-mini")
prompt_template = PromptTemplate.from_template(template = "请评价{product}的优缺点，包括{aspect1}和{aspect2}。")
prompt = prompt_template.format(product="电脑",aspect1="性能",aspect2="电池")
# prompt = prompt_template.invoke({"product":"电脑","aspect1":"性能","aspect2":"电池"})
print(type(prompt))
# llm.invoke(prompt) #使用非对话模型调用
llm1.invoke(prompt) #使用对话模型调用
```
### 4.3 具体使用：ChatPromptTemplate
#### 4.3.1 使用说明
ChatPromptTemplate是创建 聊天消息列表 的提示模板。它比普通 PromptTemplate 更适合处理多角色、多轮次的对话场景。

**特点：**
- 支持 `System/Human/AI`等不同角色的消息模板
- 对话历史维护

**参数类型:**
列表参数格式是tuple类型（`role`:str `content`:str 组合最常用）
元组的格式为：`(role: str | type, content: str | list[dict] | list[object])`
其中 `role` 是：字符串（如 `"System"`、`"Human"`、`"AI"`）
#### 4.3.2 两种实例化方式
**方式1:使用构造方法**

***举例：***
```python
from langchain_core.prompts import ChatPromptTemplate
#参数类型这里使用的是tuple构成的list
#省略了message=[],input_variables=[]
prompt_template = ChatPromptTemplate([
# 字符串 role + 字符串 content
  ("system", "你是一个AI开发工程师. 你的名字是 {name}."),
  ("human", "你能开发哪些AI应用?"),
  ("ai", "我能开发很多AI应用, 比如聊天机器人, 图像识别, 自然语言处理等."),
  ("human", "{user_input}")
])
#调用invoke()方法，返回ChatPromptValue
prompt = prompt_template.invoke(input={"name":"小谷AI","user_input":"你能帮我做什么?"})
print(type(prompt))
print(prompt)
```
**方式2:调用from_messages()**

***举例1:***
```python
# 导入相关依赖
from langchain_core.prompts import ChatPromptTemplate
# 定义聊天提示词模版
chat_template = ChatPromptTemplate.from_messages([
  ("system", "你是一个有帮助的AI机器人，你的名字是{name}。"),
  ("human", "你好，最近怎么样？"),
  ("ai", "我很好，谢谢！"),
  ("human", "{user_input}"),
])
# 格式化聊天提示词模版中的变量
messages = chat_template.invoke(input={"name":"小明", "user_input":"你叫什么名字？"})
# 打印格式化后的聊天提示词模版内容
print(messages)
```

***举例2:了解***
```python
# 示例1：role为字符串
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
  ("system", "你是一个{role}."),
  ("human", "{user_input}"),
])
# 示例2：role为消息类不支持
from langchain_core.messages import SystemMessage, HumanMessage
# prompt = ChatPromptTemplate.from_messages([
  # (SystemMessage, "你是一个{role}."), # 类对象 role + 字符串 content
  # (HumanMessage, ["你好！", {"type": "text"}]), # 类对象 role + list[dict] content
# ])
# 修改
prompt = ChatPromptTemplate.from_messages([
  ("system", ["你好！", {"type": "text"}]), # 字符串 role + list[dict] content
])
```

#### 4.3.3 模板调用的几种方式
- `invoke()`:传入的是字典，返回的是ChatPromptValue
- `format()`:传入变量的值，返回的是str
- `format_messages()`:传入变量的值，返回消息构成的list
- `format_prompt()`:传入变量的值，返回的是ChatPromptValue

**方式1:invoke()**
```python{10}
from langchain_core.prompts import ChatPromptTemplate
#参数类型这里使用的是tuple构成的list
prompt_template = ChatPromptTemplate([
# 字符串 role + 字符串 content
  ("system", "你是一个AI开发工程师. 你的名字是 {name}."),
  ("human", "你能开发哪些AI应用?"),
  ("ai", "我能开发很多AI应用, 比如聊天机器人, 图像识别, 自然语言处理等."),
  ("human", "{user_input}")
])
prompt = prompt_template.invoke({"name":"小谷AI", "user_input":"你能帮我做什么?"})
print(type(prompt))
print(prompt)
print(len(prompt.messages))
```

**方式2:format()**
```python{10}
from langchain_core.prompts import ChatPromptTemplate
#参数类型这里使用的是tuple构成的list
prompt_template = ChatPromptTemplate([
# 字符串 role + 字符串 content
  ("system", "你是一个AI开发工程师. 你的名字是 {name}."),
  ("human", "你能开发哪些AI应用?"),
  ("ai", "我能开发很多AI应用, 比如聊天机器人, 图像识别, 自然语言处理等."),
  ("human", "{user_input}")
])
prompt = prompt_template.format(name="小谷AI", user_input="你能帮我做什么?")
print(type(prompt))
print(prompt)
print(len(prompt.messages))
```
**方式3:format_messages()**
```python{10}
from langchain_core.prompts import ChatPromptTemplate
#参数类型这里使用的是tuple构成的list
prompt_template = ChatPromptTemplate([
# 字符串 role + 字符串 content
  ("system", "你是一个AI开发工程师. 你的名字是 {name}."),
  ("human", "你能开发哪些AI应用?"),
  ("ai", "我能开发很多AI应用, 比如聊天机器人, 图像识别, 自然语言处理等."),
  ("human", "{user_input}")
])
prompt = prompt_template.format_messages(name="小谷AI", user_input="你能帮我做什么?")
print(type(prompt))
print(prompt)
```
**结论**：给占位符赋值，针对于ChatPromptTemplate，推荐使用 format_messages() 方法，返回消息列表。

**方式4:format_prompt()**
```python{10}
from langchain_core.prompts import ChatPromptTemplate
#参数类型这里使用的是tuple构成的list
prompt_template = ChatPromptTemplate([
# 字符串 role + 字符串 content
  ("system", "你是一个AI开发工程师. 你的名字是 {name}."),
  ("human", "你能开发哪些AI应用?"),
  ("ai", "我能开发很多AI应用, 比如聊天机器人, 图像识别, 自然语言处理等."),
  ("human", "{user_input}")
])
prompt = prompt_template.format_prompt(name="小谷AI", user_input="你能帮我做什么?")
print(prompt.to_messages())
print(type(prompt.to_messages()))
```

#### 4.3.4 更丰富的实例化参数类型
ChatPromptTemplate的两种创建方式。无论使用构造方法，还是使用from_messages()，参数类型都是`列表类型` 。列表中的元素可以是多种类型，前面主要测试了元组类型。
源码：
```python
def __init__(self,messages: Sequence[BaseMessagePromptTemplate | BaseMessage |BaseChatPromptTemplate | tuple[str | type, str | list[dict] | list[object]] | str | dict[str, Any]],*,template_format: Literal["f-string", "mustache", "jinja2"] = "f-string",**kwargs: Any)-> None
```
源码：
```python
@classmethod 
def from_messages(cls,messages: Sequence[BaseMessagePromptTemplate |BaseMessage |BaseChatPromptTemplate | tuple[str | type, str | list[dict] | list[object]] | str | dict[str, Any]],template_format: Literal["f-string", "mustache", "jinja2"] = "f-string")-> ChatPromptTemplate
```
结论：参数是列表类型，列表的元素可以是字符串、字典、字符串构成的元组、消息类型、提示词模板类型、消息提示词模板类型等

**类型1:str类型**

不推荐！**因为默认角色都是human**
```python
#1.导入相关依赖
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage, AIMessage
# 2.定义聊天提示词模版
chat_template = ChatPromptTemplate.from_messages(
  [
  "Hello, {name}!" # 等价于 ("human", "Hello, {name}!")
  ]
)
# 3.1格式化聊天提示词模版中的变量(自己提供的)
messages = chat_template.format_messages(name="小谷AI")
# 3.2 使用invoke执行
# messages=chat_template.invoke({"name":"小谷AI"})
# 4.打印格式化后的聊天提示词模版内容
print(messages)
```

**类型2:dict类型**
```python
# 示例: 字典形式的消息
prompt = ChatPromptTemplate.from_messages([
  {"role": "system", "content": "你是一个{role}."},
  {"role": "human", "content": ["复杂内容", {"type": "text"}]},
])
print(prompt.format_messages(role="教师"))
```

**类型3:Message类型**
```python
from langchain_core.messages import SystemMessage,HumanMessage
chat_prompt_template = ChatPromptTemplate.from_messages([
  SystemMessage(content="我是一个贴心的智能助手"),
  HumanMessage(content="我的问题是:人工智能英文怎么说？")
])
messages = chat_prompt_template.format_messages()
print(messages)
print(type(messages))
```

**类型4:BaseChatPromptTemplate类型**

使用 BaseChatPromptTemplate，可以理解为ChatPromptTemplate里嵌套了ChatPromptTemplate。

***举例1:不带参数***
```python{6}
from langchain_core.prompts import ChatPromptTemplate
# 使用 BaseChatPromptTemplate（嵌套的 ChatPromptTemplate）
nested_prompt_template1 = ChatPromptTemplate.from_messages([("system", "我是一个人工智能助手")])
nested_prompt_template2 = ChatPromptTemplate.from_messages([("human", "很高兴认识你")])
prompt_template = ChatPromptTemplate.from_messages([nested_prompt_template1,nested_prompt_template2])
prompt_template.format_messages()
```
***举例2:带参数***
```python{6}
from langchain_core.prompts import ChatPromptTemplate
# 使用 BaseChatPromptTemplate（嵌套的 ChatPromptTemplate）
nested_prompt_template1 = ChatPromptTemplate.from_messages([("system", "我是一个人工智能助手，我的名字叫{name}")])
nested_prompt_template2 = ChatPromptTemplate.from_messages([("human", "很高兴认识你,我的问题是{question}")])
prompt_template = ChatPromptTemplate.from_messages([nested_prompt_template1,nested_prompt_template2])
prompt_template.format_messages(name="小智"，question="你为什么这么帅？")
```

**类型5：BaseMessagePromptTemplate类型**

LangChain提供不同类型的MessagePromptTemplate。最常用的是`SystemMessagePromptTemplate` 、`HumanMessagePromptTemplate`和`AIMessagePromptTemplate`，分别创建系统消息、人工消息和AI消息，它们是ChatMessagePromptTemplate的特定角色子类。

**基本概念：**
**HumanMessagePromptTemplate**，专用于生成`用户消息（HumanMessage）`的模板类，是ChatMessagePromptTemplate的特定角色子类。
  - 本质 ：预定义了`role="human"`的 MessagePromptTemplate，且无需无需手动指定角色
  - 模板化 ：支持使用变量占位符，可以在运行时填充具体值
  - 格式化 ：能够将模板与输入变量结合生成最终的聊天消息
  - 输出类型 ：生成 HumanMessage 对象（ content + role = "human"）
  - 设计目的：简化用户输入消息的模板化构造，避免重复定义角色

**SystemMessagePromptTemplate**、**AIMessagePromptTemplate**：类似于上面，不再赘述
**ChatMessagePromptTemplate**，用于构建聊天消息的模板。它允许你创建可重用的消息模板，这些模板可以动态地插入变量值来生成最终的聊天消息

- **角色指定**：可以为每条消息指定角色（如"system"、"human"、"ai"）等，角色灵活。
- **模板化**：支持使用变量占位符，可以在运行时填充具体值
- **格式化**：能够将模板与输入变量结合生成最终的聊天消息

***举例1:***
```python
# 导入聊天消息类模板
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate,SystemMessagePromptTemplate
# 创建消息模板
system_template = "你是一个专家{role}"
system_message_prompt = SystemMessagePromptTemplate.from_templat(system_template)
human_template = "给我解释{concept}，用浅显易懂的语言"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
# 组合成聊天提示模板
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
# 格式化提示
formatted_messages = chat_prompt.format_messages(
  role="物理学家",
  concept="相对论"
)
print(formatted_messages)
```

***举例2:ChatMessagePromptTemplate的理解***
```python
# 1.导入相关包
from langchain_core.prompts import ChatMessagePromptTemplate
# 2.定义模版
prompt = "今天我们授课的内容是{subject}"
# 3.创建自定义角色聊天消息提示词模版
chat_message_prompt = ChatMessagePromptTemplate.from_template(role="teacher", template=prompt)
# 4.格式聊天消息提示词
resp = chat_message_prompt.format(subject="我爱北京天安门")
print(type(resp))
print(resp)
```

***举例3:综合使用***
```python
from langchain_core.prompts import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage, HumanMessage
# 示例 1: 使用 BaseMessagePromptTemplate
system_prompt = SystemMessagePromptTemplate.from_template("你是一个{role}.")
human_prompt = HumanMessagePromptTemplate.from_template("{user_input}")

# 示例 2: 使用 BaseMessage（已实例化的消息）
system_msg = SystemMessage(content="你是一个AI工程师。")
human_msg = HumanMessage(content="你好！")
# 示例 3: 使用 BaseChatPromptTemplate（嵌套的 ChatPromptTemplate）
nested_prompt = ChatPromptTemplate.from_messages([("system", "嵌套提示词")])
prompt = ChatPromptTemplate.from_messages([
  system_prompt, # MessageLike (BaseMessagePromptTemplate)
  human_prompt, # MessageLike (BaseMessagePromptTemplate)
  system_msg, # MessageLike (BaseMessage)
  human_msg, # MessageLike (BaseMessage)
  nested_prompt, # MessageLike (BaseChatPromptTemplate)
])
prompt.format_messages(role="人工智能专家",user_input="介绍一下大模型的应用场景")
```
#### 4.3.5 结合LLM
***举例:***
```python
from langchain.prompts.chat import ChatPromptTemplate
######1、提供提示词#########
chat_prompt = ChatPromptTemplate.from_messages([
  ("system", "你是一个数学家，你可以计算任何算式"),
  ("human", "我的问题：{question}"),
])
# 输入提示
messages = chat_prompt.format_messages(question="我今年18岁，我的舅舅今年38岁，我的爷爷今年72岁，我和舅舅一共多少岁了？")
#print(messages)
######2、提供大模型#########
import os
import dotenv
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
chat_model = ChatOpenAI(model="gpt-4o-mini")
######3、结合提示词，调用大模型#########
# 得到模型的输出
output = chat_model.invoke(messages)
# 打印输出内容
print(output.content)
```

#### 4.3.6 插入消息列表：MessagesPlaceholder
当你不确定消息提示模板使用什么角色，或者希望在格式化过程中`插入消息列表`时，这就需要使用 `MessagesPlaceholder`，负责在特定位置添加消息列表。

**使用场景：**
多轮对话系统存储历史消息以及Agent的中间步骤处理此功能非常有用。

***举例1：***
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
prompt_template = ChatPromptTemplate.from_messages([
  ("system", "You are a helpful assistant"),
  MessagesPlaceholder("msgs")
])
# prompt_template.invoke({"msgs": [HumanMessage(content="hi!")]})
prompt_template.format_messages(msgs=[HumanMessage(content="hi!")])
```

***举例2：存储对话历史内容***
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
prompt = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{question}")
  ]
)
prompt.format_messages(
  history=[
    HumanMessage(content="1+2*3 = ?"),
    AIMessage(content="1+2*3=7")
  ],
  question="我刚才问题是什么？")
# prompt.invoke(
#   {
#     "history": [("human", "what's 5 + 2"), ("ai", "5 + 2 is 7")
#     "question": "now multiply that by 4"
#   }
# )
```

***举例3:***
```python
#1.导入相关包
from langchain_core.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,MessagesPlaceholder)
# 2.定义消息模板
prompt = ChatPromptTemplate.from_messages([
  SystemMessagePromptTemplate.from_template("你是{role}"),
  MessagesPlaceholder(variable_name="intermediate_steps"),
  HumanMessagePromptTemplate.from_template("{query}")
])
# 3.定义消息对象（运行时填充中间步骤的结果）
intermediate = [SystemMessage(name="search", content="北京: 晴, 25℃")]
# 4.格式化聊天消息提示词模版
prompt.format_messages(
  role="天气预报员",
  intermediate_steps=intermediate,
  query="北京天气怎么样？"
)
```

### 4.4 具体使用：少量样本示例的提示词模板
#### 4.4.1 使用说明
在构建prompt时，可以通过构建一个`少量示例列表`去进一步格式化prompt，这是一种简单但强大的指导生成的方式，在某些情况下可以`显著提高模型性能`。
少量示例提示模板可以由一组示例 或一个负责从定义的集合中选择`一部分示例`的示例选择器构建。
- 前者：使用`FewShotPromptTemplate`或`FewShotChatMessagePromptTemplate`
- 后者：使用`Example selectors`(示例选择器)
每个示例的结构都是一个`字典`，其中`键`是输入变量，`值`是输入变量的值。

**体会:**
zeroshot会导致低质量回答
```python
import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_BASE_URL'] = os.getenv('OPENAI_BASE_URL')
chat_model = ChatOpenAI(model="gpt-4o-mini",temperature=0.4)
res = chat_model.invoke("2 🦜 9是多少?")
print(res.content)
```
```
2 🦜 9的计算⽅式取决于你所⽤的符号“🦜”的含义。请提供更多信息或者说明这个符号代表什么运算。
```

#### 4.4.2 FewShotPromptTemplate的使用
***举例1：***
```python
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
#1、
创建示例集合
examples = [
  {"input": "北京天气怎么样", "output": "北京市"},
  {"input": "南京下雨吗", "output": "南京市"},
  {"input": "武汉热吗", "output": "武汉市"}
]
#2、创建PromptTemplate实例
example_prompt = PromptTemplate.from_template(
  template="Input: {input}\nOutput: {output}"
)
#3、创建FewShotPromptTemplate实例
prompt = FewShotPromptTemplate(
  examples=examples,
  example_prompt=example_prompt,
  suffix="Input: {input}\nOutput:", # 要放在示例后面的提示模板字符
  input_variables=["input"] # 传入的变量
)
#4、调用
prompt = prompt.invoke({"input":"长沙多少度"})
print("===Prompt===")
print(prompt)
```
```
===Prompt===
Input: 北京天气怎么样
Output: 北京市
Input: 南京下雨吗
Output: 南京市
Input: 武汉热吗
Output: 武汉市
Input: 长沙多少度
Output:
```

**结合大模型调用：**
```python
import os
import dotenv
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
#获取大模型
chat_model = ChatOpenAI(model="gpt-4o-mini")
#调用
print("===Response===")
response = chat_model.invoke(prompt)
print(response.content)
```
```
===Response===
长沙市
```

#### 4.4.3 FewShotChatMessagePromptTemplate的使用
除了FewShotPromptTemplate之外，FewShotChatMessagePromptTemplate是专门为`聊天对话场景`设计的少样本（few-shot）提示模板，它继承自 `FewShotPromptTemplate`，但针对聊天消息的格式进行了优化。

**特点：**
- 自动将示例格式化为聊天消息（`HumanMessage`/`AIMessage`等）
- 输出结构化聊天消息（`List[BaseMessage]`）
- 保留对话轮次结构

***举例1：基本结构***
```python
from langchain.prompts import (FewShotChatMessagePromptTemplate,ChatPromptTemplate)
# 1.示例消息格式
examples = [
  {"input": "1+1等于几？", "output": "1+1等于2"},
  {"input": "法国的首都是？", "output": "巴黎"}
]
# 2.定义示例的消息格式提示词模版
msg_example_prompt = ChatPromptTemplate.from_messages([
  ("human", "{input}"),
  ("ai", "{output}"),
])
# 3.定义FewShotChatMessagePromptTemplate对象
few_shot_prompt = FewShotChatMessagePromptTemplate(
  example_prompt=msg_example_prompt,
  examples=examples
)
# 4.输出格式化后的消息
print(few_shot_prompt.format())
```
```
Human: 1+1等于几？
AI: 1+1等于2
Human: 法国的首都是？
AI: 巴黎
```

***举例2：使用方式：将原始输入和被选中的示例组一起加入Chat提示词模版中。***
```python
# 1.导入相关包
from langchain_core.prompts import (FewShotChatMessagePromptTemplate,ChatPromptTemplate)
# 2.定义示例组
examples = [
  {"input": "2🦜2", "output": "4"},
  {"input": "2🦜3", "output": "8"},
]
# 3.定义示例的消息格式提示词模版
example_prompt = ChatPromptTemplate.from_messages([
  ('human','{input} 是多少?'),
  ('ai','{output}')
])
# 4.定义FewShotChatMessagePromptTemplate对象
few_shot_prompt = FewShotChatMessagePromptTemplate(
  examples=examples, # 示例组
  example_prompt=example_prompt, # 示例提示词词模版
)
# 5.输出完整提示词的消息模版
final_prompt = ChatPromptTemplate.from_messages(
  [
    ('system','你是一个数学奇才'),
    few_shot_prompt,
    ('human','{input}'),
  ]
)
#6.提供大模型
import os
import dotenv
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
chat_model = ChatOpenAI(model="gpt-4o-mini",temperature=0.4)
chat_model.invoke(final_prompt.invoke(input="2🦜4")).content
```
```
'2🦜4 等于 16。'
```
#### 4.4.4 Example selectors(示例选择器)
前面FewShotPromptTemplate的特点是，无论输入什么问题，都会包含全部示例。在实际开发中，我们可以根据当前输入，使用示例选择器，从大量候选示例中选取最相关的示例子集。

**使用的好处:**
避免盲目传递所有示例，减少 token 消耗的同时，还可以提升输出效果。

**示例选择策略:** 
语义相似选择、长度选择、最大边际相关示例选择等
- **语义相似选择:** 通过余弦相似度等度量方式评估语义相关性，选择与输入问题最相似的`k`个示例。
- **长度选择:** 根据输入文本的长度，从候选示例中筛选出长度最匹配的示例。增强模型对文本结构的理解。比语义相似度计算更轻量，适合对响应速度要求高的场景。

- **最大边际相关示例选择:** 优先选择与输入问题语义相似的示例；同时，通过惩罚机制避免返回同质化的内容

:::info
- 余弦相似度是通过计算两个向量的夹⻆余弦值来衡量它们的相似性。它的值范围在-1到1之间：当两个向量⽅向相同时值为1；夹⻆为90°时值为0；⽅向完全相反时为-1。
- 数学表达式：余弦相似度 = (A·B)/ (||A|| * ||B||)。其中A·B是点积，||A||和||B||是向量的模（⻓度）。
:::
***举例1：***
```python
# 1.导入相关包
import os
import dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchian_openai import OpenAIEmbeddings
dotenv.load_dotenv()
# 2.定义嵌入模型
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# 3.定义示例组
examples = [
  {
    "question": "谁活得更久，穆罕默德·阿里还是艾伦·图灵?",
    "answer": """
    接下来还需要问什么问题吗？
    追问：穆罕默德·阿里去世时多大年纪？
    中间答案：穆罕默德·阿里去世时享年74岁。
    """,
  },
  {
    "question": "craigslist的创始人是什么时候出生的？",
    "answer": """
    接下来还需要问什么问题吗？
    追问：谁是craigslist的创始人？
    中级答案：Craigslist是由克雷格·纽马克创立的。
    """,
  },
  {
    "question": "谁是乔治·华盛顿的外祖父？",
    "answer": """
    接下来还需要问什么问题吗？
    追问：谁是乔治·华盛顿的母亲？
    中间答案：乔治·华盛顿的母亲是玛丽·鲍尔·华盛顿。
    """,
  },
  {
    "question": "《大白鲨》和《皇家赌场》的导演都来自同一个国家吗？",
    "answer": """
    接下来还需要问什么问题吗？
    追问：《大白鲨》的导演是谁？
    中级答案：《大白鲨》的导演是史蒂文·斯皮尔伯格。
    """,
  },
]

# 4.定义示例选择器
example_selector = SemanticSimilarityExampleSelector.from_examples(
  # 这是可供选择的示例列表
  examples,
  # 这是用于生成嵌入的嵌入类，用于衡量语义相似性
  embeddings_model,
  # 这是用于存储嵌入并进行相似性搜索的 VectorStore 类
  Chroma,
  # 这是要生成的示例数量
  k=1,
)
# 选择与输入最相似的示例
question = "玛丽·鲍尔·华盛顿的父亲是谁?"
selected_examples = example_selector.select_examples({"question": question})
print(f"与输入最相似的示例：{selected_examples}")
# for example in selected_examples:
# print("\n")
# for k, v in example.items():
# print(f"{k}: {v}")
```
```
question: 谁是乔治·华盛顿的外祖⽗？
answer:
接下来还需要问什么问题吗？
追问：谁是乔治·华盛顿的⺟亲？
中间答案：乔治·华盛顿的⺟亲是玛丽·鲍尔·华盛顿
```

***举例2:结合FewShotPromptTemplate 使用***
这里使用FAISS，需安装：
```python
pip install faiss-cpu
#或
conda install faiss-cpu
```
```python
# 1.导入相关包
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import OpenAIEmbeddings
# 2.定义示例提示词模版
example_prompt = PromptTemplate.from_template(
  template="Input: {input}\nOutput: {output}",
)
# 3.创建一个示例提示词模版
examples = [
  {"input": "高兴", "output": "悲伤"},
  {"input": "高", "output": "矮"},
  {"input": "长", "output": "短"},
  {"input": "精力充沛", "output": "无精打采"},
  {"input": "阳光", "output": "阴暗"},
  {"input": "粗糙", "output": "光滑"},
  {"input": "干燥", "output": "潮湿"},
  {"input": "富裕", "output": "贫穷"},
]
)
# 4.定义嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# 5.创建语义相似性示例选择器
example_selector = SemanticSimilarityExampleSelector.from_examples(
  examples,
  embeddings,
  FAISS,
  k=2,
)
#或者
#example_selector = SemanticSimilarityExampleSelector(
# examples,
# embeddings,
# FAISS,
# k=2
#)
# 6.定义小样本提示词模版
similar_prompt = FewShotPromptTemplate(
  example_selector=example_selector,
  example_prompt=example_prompt,
  prefix="给出每个词组的反义词",
  suffix="Input: {word}\nOutput:",
  input_variables=["word"],
)
response = similar_prompt.invoke({"word":"忧郁"})
print(response.text)
```
```
给出每个词组的反义词
Input: ⾼兴
Output: 悲伤
Input: 阳光
Output: 阴暗
Input: 忧郁
Output
```
### 4.5 具体使用：PipelinePromptTemplate(了解)
用于将多个提示模板**按顺序组合成处理管道**，实现分阶段、模块化的提示构建。它的核心作用类似于软件开发中的**管道模式**（Pipeline Pattern），通过串联多个提示处理步骤，实现复杂的提示生成逻辑。

**特点：**
- 将复杂提示拆解为多个处理阶段，每个阶段使用独立的提示模板
- 前一个模板的输出作为下一个模板的输入变量
- 使用场景：解决单一超大提示模板难以维护的问题

**说明：**
PipelinePromptTemplate在langchain 0.3.22版本中被标记为过时，在langchain-core==1.0之前不会删除它。

***举例：***
```python
from langchain_core.prompts.pipeline import PipelinePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
# 阶段1：问题分析
analysis_template = PromptTemplate.from_template("""
分析这个问题：{question}
关键要素：
""")

# 阶段2：知识检索
retrieval_template = PromptTemplate.from_template("""
基于以下要素搜索资料：
{analysis_result}
搜索关键词：
""")

# 阶段3：生成最终回答
answer_template = PromptTemplate.from_template("""
综合以下信息回答问题：
{retrieval_result}
最终答案：
""")

# 构建管道
pipeline = PipelinePromptTemplate(
  final_prompt=answer_template,
  pipeline_prompts=[
    ("analysis_result", analysis_template),
    ("retrieval_result", retrieval_template)
  ]
)
print(pipeline.format(question="量子计算的优势是什么？"))
```
```
综合以下信息回答问题：
基于以下要素搜索资料：
分析这个问题：量⼦计算的优势是什么？
关键要素：
搜索关键词：
最终答案：
```
> 上述代码执行时，提示PipelinePromptTemplate已过时，代码更新如下：
```python
from langchain_core.prompts.prompt import PromptTemplate
# 阶段1：问题分析
analysis_template = PromptTemplate.from_template("""
分析这个问题：{question}
关键要素：
""")
# 阶段2：知识检索
retrieval_template = PromptTemplate.from_template("""
基于以下要素搜索资料：
{analysis_result}
搜索关键词：
""")

# 阶段3：生成最终回答
answer_template = PromptTemplate.from_template("""
综合以下信息回答问题：
{retrieval_result}
最终答案：
""")

# 逐步执行管道提示
pipeline_prompts = [
  ("analysis_result", analysis_template),
  ("retrieval_result", retrieval_template)
]

my_input = {"question": "量子计算的优势是什么？"}
# print(pipeline_prompts)
# [('analysis_result', PromptTemplate(input_variables=['question'], input_types={},partial_variables={}, template='\n分析这个问题：{question}\n关键要素：\n')), ('retrieval_result',PromptTemplate(input_variables=['analysis_result'], input_types={}, partial_variables={},template='\n基于以下要素搜索资料：\n{analysis_result}\n搜索关键词：\n'))]
for name, prompt in pipeline_prompts:
# 调用当前提示模板并获取字符串结果
result = prompt.invoke(my_input).to_string()
# 将结果添加到输入字典中供下一步使用
my_input[name] = result

# 生成最终答案
my_output = answer_template.invoke(my_input).to_string()
print(my_output)
```

### 4.6 具体使用：自定义提示词模版(了解)
> 创建prompt时，可创建自定义提示模版。

**步骤：**
- 自定义类继承提示词基类模版`BasePromptTemplate`
- 重写`format`、`format_prompt`、`from_template`方法

***举例：***
```python
# 1.导入相关包
from typing import List, Dict, Any
from langchain.prompts import BasePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.schema import PromptValue
# 2.自定义提示词模版
class SimpleCustomPrompt(BasePromptTemplate):
"""简单自定义提示词模板"""
template: str
def __init__(self, template: str,**kwargs):
  # 使用PromptTemplate解析输入变量
  prompt = PromptTemplate.from_template(template)
  super().__init__(
    input_variables=prompt.input_variables,
    template=template,
    **kwargs
  )
def format(self,**kwargs: Any) -> str:
  """格式化提示词"""
  # print("kwargs:", kwargs)
  # print("self.template:", self.template)
  return self.template.format(**kwargs)

def format_prompt(self,**kwargs: Any) -> PromptValue:
  """实现抽象方法"""
  return PromptValue(text=self.format(**kwargs))

@classmethod
def from_template(cls, template: str,)-> "SimpleCustomPrompt":
  """从模板创建实例"""
  return cls(template=template,**kwargs)
# 3.使用自定义提示词模版
custom_prompt = SimpleCustomPrompt.from_template(
  template="请回答关于{subject}的问题：{question}"
)

# 4.格式化提示词
formatted = custom_prompt.format(
  subject="人工智能",
  question="什么是LLM？"
)
print(formatted)
```
```
请回答关于⼈⼯智能的问题：什么是LLM？
```

### 4.7 从文档中加载Prompt（了解）
一方面，将想要设定prompt所支持的格式保存为JSON或者YAML格式文件。另一方面，通过读取指定路径的格式化文件，获取相应的prompt。
目的与使用场景：
- 为了便于共享、存储和加强对prompt的版本控制。
- 当我们的prompt模板数据较大时，我们可以使用外部导入的方式进行管理和维护。

#### 4.7.1 yaml格式提示词
> asset下创建yaml文件：`prompt.yaml`

```yaml
_type:
  "prompt"
input_variables:
  ["name","what"]
template:
  "请给{name}讲一个关于{what}的故事"
```

```python
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv

load_dotenv()

prompt = load_prompt("asset/prompt.yaml", encoding="utf-8")
# print(prompt)
print(prompt.format(name="年轻人", what="滑稽"))
```
```
请给年轻⼈讲⼀个关于滑稽的笑话
```

#### 4.8.2 json格式提示词
> asset下创建json文件：`prompt.json`
```json
{
  "_type": "prompt",
  "input_variables": ["name", "what"],
  "template": "请{name}讲一个{what}的故事。"
}
```
**代码：**

```python
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv

load_dotenv()

prompt = load_prompt("asset/prompt.json",encoding="utf-8")
print(prompt.format(name="张三",what="搞笑的"))
```
```
请张三讲⼀个搞笑的的故事。
```

## 5. Model I/O之Output Parsers
语言模型返回的内容通常都是字符串的格式（文本格式），但在实际AI应用开发过程中，往往希望
model可以返回**更直观、更格式化的内容**，以确保应用能够顺利进行后续的逻辑处理。此时，
LangChain提供的`输出解析器`就派上用场了。
输出解析器（Output Parser）负责获取`LLM`的输出并将其转换为更合适的格式。这在**应用开发中及其重要**。

### 5.1 输出解析器的分类
LangChain有许多不同类型的输出解析器
- **`StrOutputParser`** ：字符串解析器
- **`JsonOutputParser`**：JSON解析器，确保输出符合特定JSON对象格式
- **`XMLOutputParser`**：XML解析器，允许以流行的XML格式从LLM获取结果
- **`CommaSeparatedListOutputParser`**：CSV解析器，模型的输出以逗号分隔，以列表形式返回输出
- **`DatetimeOutputParser`**：日期时间解析器，可用于将 LLM 输出解析为日期时间格式

除了上述常用的输出解析器之外，还有：
- **`EnumOutputParser`**：枚举解析器，将LLM的输出，解析为预定义的枚举值
- **`StructuredOutputParser`**：将非结构化文本转换为预定义格式的结构化数据（如字典）
- **`OutputFixingParser`**：输出修复解析器，用于自动修复格式错误的解析器，比如将返回的不符合
预期格式的输出，尝试修正为正确的结构化数据（如 JSON）
- **`RetryOutputParser`**：重试解析器，当主解析器（如 JSONOutputParser）因格式错误无法解析
LLM 的输出时，通过调用另一个 LLM 自动修正错误，并重新尝试解析

### 5.2 具体解析器的使用
**① 字符串解析器 StrOutputParser**

StrOutputParser 简单地将`任何输入`转换为`字符串`。它是一个简单的解析器，从结果中提取content字段
举例：将一个对话模型的输出结果，解析为字符串输出
```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
import os
import dotenv
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")

chat_model = ChatOpenAI(model="gpt-4o-mini")

messages = [
  SystemMessage(content="将以下内容从英语翻译成中文"),
  HumanMessage(content="It's a nice day today"),
]

# result = chat_model.invoke(messages)
# print(type(result))
# print(result)

parser = StrOutputParser()
#使用parser处理model返回的结果
response = parser.invoke(result)
print(type(response))
print(response)
```

**② JSON解析器 JsonOutputParser**

JsonOutputParser，即JSON输出解析器，是一种用于将大模型的`自由文本输出`转换为`结构化JSON数据`的工具。

**适合场景：**
特别适用于需要严格结构化输出的场景，比如 API 调用、数据存储或下游任务处理。

**实现方式**
- 方式1：用户自己通过提示词指明返回Json格式
- 方式2：借助JsonOutputParser的`get_format_instructions()`，生成格式说明，指导模型输出JSON结构

***举例1：***
```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

chat_model = ChatOpenAI(model="gpt-4o-mini")

chat_prompt_template = ChatPromptTemplate.from_messages([
  ("system","你是一个靠谱的{role}"),
  ("human","{question}")
])
parser = JsonOutputParser()

# 方式1：
result = chat_model.invoke(chat_prompt_template.format_messages(role="人工智能专家",question="人工智能用英文怎么说？问题用q表示，答案用a表示，返回一个JSON格式"))
print(result)
print(type(result))

parser.invoke(result)

# 方式2：
# chain = chat_prompt_template | chat_model | parser
# chain.invoke({"role":"人工智能专家","question" : "人工智能用英文怎么说？问题用q表示，答案用a表示，返回一个JSON格式"})
```

***举例2：***
```python
# 引入依赖包
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

chat_model = ChatOpenAI(model="gpt-4o-mini")

joke_query = "告诉我一个笑话。"
# 定义Json解析器
parser = JsonOutputParser()
# 定义提示词模版
# 注意，提示词模板中需要部分格式化解析器的格式要求format_instructions
prompt = PromptTemplate(
template="回答用户的查询.\n{format_instructions}\n{query}\n",
input_variables=["query"],
partial_variables={"format_instructions": parser.get_format_instructions()},
)
# 5.使用LCEL语法组合一个简单的链
chain = prompt | chat_model | parser
# 6.执行链
output = chain.invoke({"query": "给我讲一个笑话"})
print(output)
```
```
{'joke': '为什么海洋总是咸的？因为它有太多的"海"湿的事情发⽣！'}
```

**③ XML解析器 XMLOutputParser**
XMLOutputParser，将模型的自由文本输出转换为可编程处理的XML数据。
**如何实现：**在 PromptTemplate 中指定XML格式要求，让模型返回<tag>content</tag> 形式的数据。

**注意：** XMLOutputParser不会直接将模型的输出保持为原始XML字符串，而是会解析XML并转换成**Python字典**（或类似结构化的数据）。目的是为了方便程序后续处理数据，而不是单纯保留XML格式。

***举例：***
```python
# 1.导入相关包
from langchain_core.output_parsers import XMLOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
# 2. 初始化语言模型
chat_model = ChatOpenAI(model="gpt-4o-mini")
# 3.测试模型的xml解析效果
actor_query = "生成汤姆·汉克斯的简短电影记录,使用中文回复"
# 4.定义XMLOutputParser对象
parser = XMLOutputParser()
# 5.定义提示词模版对象
# prompt = PromptTemplate(
# template="{query}\n{format_instructions}",
# input_variables=["query","format_instructions"],
# partial_variables={"format_instructions": parser.get_format_instructions()},
#)
prompt_template = PromptTemplate.from_template("{query}\n{format_instructions}")
prompt_template1 =prompt_template.partial(format_instructions=parser.get
_format_instructions())

response = chat_model.invoke(prompt_template1.format(query=actor_query))
print(response.content)

# 方式1
response = chat_model.invoke(prompt_template1.format(query=actor_query))
result = parser.invoke(response)
print(result)
print(type(result))
# 方式2
# chain = prompt_template1 | chat_model | parser
# result = chain.invoke({"query":actor_query})
# print(result)
# print(type(result))
```
**④ 列表解析器 CommaSeparatedListOutputParser**
列表解析器：利用此解析器可以将模型的文本响应转换为一个用`逗号分隔的列表（List[str]）`。
```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()
# 返回一些指令或模板，这些指令告诉系统如何解析或格式化输出数据
format_instructions = output_parser.get_format_instructions()
print(format_instructions)

messages = "大象,猩猩,狮子"
result = output_parser.parse(messages)
print(result)
print(type(result))
```
```
Your response should be a list of comma separated values, eg: foo, bar, baz or foo,bar,baz
['⼤象', '猩猩', '狮⼦']
<class 'list'>
```
**⑤ 日期解析器 DatetimeOutputParser (了解)**
利用此解析器可以直接将LLM输出解析为日期时间格式。
- **get_format_instructions()**： 获取日期解析的格式化指令，指令为："Write a datetime string that matches the following pattern: '%Y-%m-%dT%H:%M:%S.%fZ'。
  - **举例：**
  1206-08-16T17:39:06.176399Z

***举例：***
```python
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import DatetimeOutputParser

chat_model = ChatOpenAI(model="gpt-4o-mini")

chat_prompt = ChatPromptTemplate.from_messages([
  ("system","{format_instructions}"),
  ("human", "{request}")
])

output_parser = DatetimeOutputParser()

# 方式1：
# model_request = chat_prompt.format_messages(
# request="中华人民共和国是什么时候成立的",
# format_instructions=output_parser.get_format_instructions()
# )
# response = chat_model.invoke(model_request)
# result = output_parser.invoke(response)
# print(result)
# print(type(result))
# 方式2：
chain = chat_prompt | chat_model | output_parser

resp = chain.invoke({"request":"中华人民共和国是什么时候成立的","format_instructions":output_parser.get_format_instructions()})

print(resp)
print(type(resp))
```
```
1949-10-01 00:00:00
<class 'datetime.datetime'>
```

