# LangChain Agents
## 1. 理解Agents
通用人工智能（AGI）将是AI的终极形态，几乎已成为业界共识。同样，构建智能体（Agent）则是AI工程应用当下的“终极形态”。

### 1.1 Agent与Chain的区别
在Chain中行动序列是`硬编码的、固定流程的`，像是“线性流水线”，而Agent则采用语言模型作为`推理引擎`，具备一定的`自主决策`能力，来确定以什么样的顺序采取什么样的行动，像是“拥有大脑的机器工人”。

它可以根据任务`动态决定`：
- 如何**拆解任务**
- 需要**调用哪些工具**
- 以**什么顺序调用**
- 如何利用好 `中间结果` 推进任务

### 1.2 什么是Agent
Agent（智能体） 是一个通过动态协调`大语言模型（LLM）`和`工具（Tools）`来完成复杂任务的智能系统。它让LLM充当"决策大脑"，根据用户输入自主选择和执行工具（如搜索、计算、数据库查询等），最终生成精准的响应。

### 1.3 Agent的核心能力/组件
作为一个智能体，需要具备以下核心能力：

![alt text](/public/langchain/agent/1.png)

1）**大模型(LLM)**：作为大脑，提供推理、规划和知识理解能力。
- 比如：OpenaAI()、ChatOpenAI()

2）**记忆(Memory)**：具备短期记忆（上下文）和长期记忆（向量存储），支持快速知识检索。
- 比如：ConversationBufferMemory、ConversationSummaryMemory、ConversationBufferWindowMemory等

3）**工具(Tools)**：调用外部工具（如API、数据库）的执行单元
- 比如：SearchTool、CalculatorTool

4）**规划(Planning)**：任务分解、反思与自省框架实现复杂任务处理

5）**行动(Action)**：实际执行决策的能力
- 比如：检索、推理、编程

6）**协作:** 通过与其他智能体交互合作，完成更复杂的任务目标。

**问题：** 为什么要调用第三方工具（比如：搜索引擎或者 数据库）或借助第三方库呢？
因为大模型虽然非常强大，但是也具备一定的局限性。比如不能回答 实时信息 、处理 复杂数学逻辑问题，仍然非常初级等等。因此，可以借助第三方工具来辅助大模型的应用。

以MCP工具为例说明：https://bailian.console.aliyun.com/?tab=mcp#/mcp-market

### 1.5 明确几个组件
Agents 模块有几个关键组件：

**1、工具 Tool**
LangChain 提供了广泛的入门工具，但也支持 `自定义工具` ，包括自定义描述。

在框架内，每个功能或函数被 `封装成一个工具` （Tools），具有自己的输入、输出及处理方法。

具体使用步骤：

① Agent 接收任务后，通过大模型推理选择适合的工具处理任务。

② 一旦选定，LangChain将任务输入传递给该工具，工具处理输入生成输出。

③ 输出经过大模型推理，可用于其他工具的输入或作为最终结果返回给用户。

**2、工具集 Toolkits**
在构建Agent时，通常提供给LLM的工具不仅仅只有一两个，而是一组可供选择的工具集(Tool列表)，这样可以让 LLM 在完成任务时有更多的选择。

**3、智能体/代理 Agent**
智能体/代理（agent）可以协助我们做出决策，调用相应的 API。底层的实现方式是通过 LLM 来决定下一步执行什么动作。

**4、代理执行器 AgentExecutor**
AgentExecutor本质上是代理的运行时，负责协调智能体的决策和实际的工具执行。
```
AgentExecutor是⼀个很好的起点，但是当你开始拥有更多定制化的代理时，它就不够灵活了。为了解决这个问题，我们构建了LangGraph，使其成为这种灵活、⾼度可控的运⾏时。
```

## 2. Agent 入门使用
### 2.1 Agent、AgentExecutor的创建

| |环节1：创建Agent | 环节2：创建AgentExecutor |
| :----: | :----: | :----: |
| 方式1：传统方式 | 使用 AgentType | 指定 initialize_agent() |
| 方式2：通用方式 | create_xxx_agent() 比如：create_react_agent()、create_tool_calling_agent() | 调用AgentExecutor()构造方法|

### 2.2 Agent的类型
> 顾名思义就是某件事可以由不同的⼈去完成，最终结果可能是⼀样的，但是做的过程可能各有千秋。⽐如⼀个公司需求， 普通开发 可以编写， 技术经理 也可以编写， CTO 也可以编写。虽然都能完成最后的需求，但是CTO做的过程可能更加直观，⾼效。

在LangChain中Agent的类型就是为你提供不同的"问题解决姿势"的。
API说明：https://python.langchain.com/v0.1/docs/modules/agents/agent_types/

Agents的核心类型有两种模式：
- 方式1：Funcation Call模式
- 方式2：ReAct 模式

#### 2.2.1 FUNCATION_CALL模式
基于 `结构化函数调用` （如 OpenAI Function Calling）
直接生成工具调用参数（ `JSON 格式` ）
效率更高，适合工具明确的场景

**典型 AgentType：**

```python
#第1种：
AgentType.OPENAI_FUNCTIONS
#第2种：
AgentType.OPENAI_MULTI_FUNCTIONS
```

**工作流程示例：**

```
第1步：找到Search工具：{"tool": "Search","args": {"query": "LangChain最新版本"}}
第2步：执行Search工具
======================================
第1步：找打scrape_website工具：{"tool": "Search","args": {"target": "LangChain最新版本","url":"要抓取的网站地址"}}
第2步：执行scrape_website工具
```
#### 2.2.2 ReAct 模式
- 基于 `文本推理` 的链式思考（Reasoning + Acting），具备反思和自我纠错能力。
  - 推理（Reasoning）：分析当前状态，决定下一步行动
  - 行动（Acting）：调用工具并返回结果
- 通过 `自然语言描述决策过程`
- 适合需要明确推理步骤的场景。例如智能客服、问答系统、任务执行等。

**典型 AgentType：**
```python
#第1种：零样本推理(可以在没有预先训练的情况下尝试解决新的问题)
AgentType.ZERO_SHOT_REACT_DESCRIPTION
#第2种：无记忆对话
AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
#第3种：带记忆对话
AgentType.CONVERSATIONAL_REACT_DESCRIPTION
```
**工作流程示例：**
问题：我想要查询xxx
思考：我需要先搜索最新信息 → 行动：调用Search工具 → 观察：获得3个结果 →
思考：需要抓取第一个链接 → 行动：调用scrape_website工具...→ 观察：获得工具结果
最后：获取结果

**Agent两种典型类型对比表**

|特性 |Function Call模式| ReAct 模式|
|:----:|:----:|:----:|
|底层机制 |结构化函数调用| 自然语言推理|
|输出格式 |JSON/结构化数据 |自由文本|
|适合场景 |需要高效工具调用 |需要解释决策过程|
|典型延迟 |较低 （直接参数化调用）| 较高 （需生成完整文本）|
|LLM要求 |需支持函数调用（如gpt-4）| 通用模型即可|

### 2.3 AgentExecutor创建方式

**传统方式：initialize_agent()**

- **特点：**
  - 内置一些标准化模板（如 ZERO_Agent的创建：使用AgentType_SHOT_REACT_DESCRIPTION
  - 优点：快速上手（3行代码完成配置）
  - 缺点：定制化能力较弱（如提示词固定）

- **代码片段：**
```python
from langchain.agents import initialize_agent

#第1步：创建AgentExecutor
agent_executor = initialize_agent(
  llm=llm,
  agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
  tools=[search_tool],
  verbose=True
)
#第2步：执行
agent_executor.invoke({"xxxx"})
```

**通用方式：AgentExecutor构造方法**

- **特点：**
  - Agent的创建：使用create_xxx_agent
- 优点：
  - 可自定义提示词（如从远程hub获取或本地自定义）
  - 清晰分离Agent逻辑与执行逻辑
- 缺点：
  - 需要更多代码
  - 需理解底层组件关系

代码片段：
```python
prompt = hub.pull("hwchase17/react")
tools = [search_tool]
#第1步：创建Agent实例
agent = create_react_agent(
  llm=llm,
  prompt=prompt,
  tools=tools
)
#第2步：创建AgentExecutor实例
agent_executor = AgentExecutor(
agent=agent,
tools=tools
)
#第3步：执行
agent_executor.invoke({"input":"xxxxx"})
```

### 2.4 小结创建方式

|组件 |传统方式| 通用方式|
|:----:|:----:|:----:|
|Agent创建| 通过 AgentType 枚举选择预设 |通过 create_xxx_agen显式构建|
|AgentExecutor创建 | 通过 initialize_agent()创建 |通过 AgentExecutor() 创建 |
|提示词 |内置不可见 |可以自定义|
|工具集成| AgentExecutor中显式传入 |Agent/AgentExecutor中需显式传入|

## 3. Agent中工具的使用
### 3.1 传统方式

**案例1：单工具使用**

- 需求：今天北京的天气怎么样?
- 使用Tavily搜索工具
  - Tavily的搜索API是一个专门为人工智能Agent(或LLM)构建的搜索引擎，可以快速提供实时、准确和真实的结果。
  - LangChain 中有一个内置工具，可以轻松使用 Tavily 搜索引擎 作为工具。
  - TAVILY_API_KEY申请：https://tavily.com/ 注册账号并登录，创建 API 密钥。

**方式1：ReAct模式**

- AgentType是 `ZERO_SHOT_REACT_DESCRIPTION`
```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# 1. 设置 API 密钥
os.environ["TAVILY_API_KEY"] = "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"

# 2. 初始化搜索工具
search = TavilySearchResults(max_results=3)

# 3. 创建Tool的实例 （本步骤可以考虑省略，直接使用[search]替换[search_tool]。但建议加上
search_tool = Tool(
  name="Search",
  func=search.run,
  description="用于搜索互联网上的信息"
)
# 4. 初始化 LLM
dotenv.load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
llm = ChatOpenAI(
  model="gpt-4o-mini",
  temperature=0,
)
# 5. 创建 AgentExecutor
agent_executor = initialize_agent(
  tools=[search_tool],
  llm=llm,
  agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
  verbose=True
)
# 5. 测试查询
query = "今天北京的天气怎么样？"
result = agent_executor.invoke(query)
print(f"查询结果: {result}")
```

**方式2：FUNCATION_CALL模式**
- AgentType是 `OPENAI_FUNCTIONS`

提示：只需要修改前面代码中的initialize_agent中的agent参数值。

```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# 1. 设置 API 密钥
os.environ["TAVILY_API_KEY"] = "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"

# 2. 初始化搜索工具
search = TavilySearchResults(max_results=3)

# 3. 创建Tool的实例 （本步骤可以考虑省略，直接使用[search]替换[search_tool]。但建议加上
search_tool = Tool(
  name="Search",
  func=search.run,
  description="用于搜索互联网上的信息"
)
# 4. 初始化 LLM
dotenv.load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
llm = ChatOpenAI(
  model="gpt-4o-mini",
  temperature=0,
)
# 5. 创建 AgentExecutor
agent_executor = initialize_agent(
  tools=[search_tool],
  llm=llm,
  agent=AgentType.OPENAI_FUNCTIONS, #唯一变化
  verbose=True
)
# 5. 测试查询
query = "今天北京的天气怎么样？"
result = agent_executor.invoke(query)
print(f"查询结果: {result}")
```

**二者对比：ZERO_SHOT_REACT_DESCRIPTION和OPENAI_FUNCTIONS**

|对比维度| ZERO_SHOT_REACT_DESCRIPTION |OPENAI_FUNCTIONS|
|:----:|:----:|:----:|
|底层机制 |模型生成文本指令，系统解析后调用工具|模型直接返回JSON格式工具调用|
|执行效率 |🐢 较低（需多轮文本交互）| ⚡ 更高（单步完成）|
|输出可读性| 直接显示人类可读的思考过程 |需查看结构化日志|
|工具参数处理| 依赖模型文本描述准确性 |自动匹配工具参数结构|
|兼容模型 |所有文本生成模型| 仅GPT-4/Claude 3等新模型|
|复杂任务表现 |可能因文本解析失败出错| 更可靠（结构化保证）|

**案例2：多工具使用**

- 需求：
  - 计算特斯拉当前股价是多少？
  - 比去年上涨了百分之几？（提示：调用PythonREPL实例的run方法）
- 多个（两个）工具的选择

**方式1：ReAct 模式**
- AgentType是 `ZERO_SHOT_REACT_DESCRIPTION`

```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_experimental.utilities.python import PythonREPL
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# 1. 设置 API 密钥
os.environ["TAVILY_API_KEY"] = "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"

# 2. 初始化搜索工具
search = TavilySearchResults(max_results=3)

# 3. 创建Tool的实例 （本步骤可以考虑省略，直接使用[search]替换[search_tool]。但建议加上
search_tool = Tool(
  name="Search",
  func=search.run,
  description="用于搜索互联网上的信息,特别是股票价格和新闻"
)

# 4.定义计算工具
python_repl = PythonREPL() # LangChain封装的工具类可以进行数学计算

calc_tool = Tool(
  name="Calculator",
  func=python_repl.run,
  description="用于执行数学计算，例如计算百分比变化"
)

# 5. 定义LLM
llm = ChatOpenAI(
  model="gpt-4o-mini",
  temperature=0,
)
# 6. 创建 AgentExecutor
agent_executor = initialize_agent(
  tools=[search_tool,calc_tool],
  llm=llm,
  agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
  verbose=True
)
# 7. 测试查询
query = "特斯拉当前股价是多少？比去年上涨了百分之几？"
result = agent_executor.invoke(query)
print(f"查询结果: {result}")
```

**方式2：FUNCATION_CALL模式**
- AgentType是 `OPENAI_FUNCTIONS`

```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_experimental.utilities.python import PythonREPL
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# 1. 设置 API 密钥
os.environ["TAVILY_API_KEY"] = "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"

# 2. 初始化搜索工具
search = TavilySearchResults(max_results=3)

# 3. 创建Tool的实例 （本步骤可以考虑省略，直接使用[search]替换[search_tool]。但建议加上
search_tool = Tool(
  name="Search",
  func=search.run,
  description="用于搜索互联网上的信息,特别是股票价格和新闻"
)

# 4.定义计算工具
python_repl = PythonREPL() # LangChain封装的工具类可以进行数学计算

calc_tool = Tool(
  name="Calculator",
  func=python_repl.run,
  description="用于执行数学计算，例如计算百分比变化"
)

# 5. 定义LLM
llm = ChatOpenAI(
  model="gpt-4o-mini",
  temperature=0,
)
# 6. 创建 AgentExecutor
agent_executor = initialize_agent(
  tools=[search_tool,calc_tool],
  llm=llm,
  agent=AgentType.OPENAI_FUNCTIONS, #唯一变化
  verbose=True
)
# 7. 测试查询
query = "特斯拉当前股价是多少？比去年上涨了百分之几？"
result = agent_executor.invoke(query)
print(f"查询结果: {result}")
```

**案例3：自定义函数与工具**

需求：计算3的平方，Agent自动调用工具完成

```python
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_openai import ChatOpenAI
import langchain
# 1. 定义工具 - 计算器（要求字符串输入）
def simple_calculator(expression: str) -> str:
  """
  基础数学计算工具，支持加减乘除和幂运算
  参数:
    expression: 数学表达式字符串，如 "3+5" 或 "2**3"
  返回:
    计算结果字符串或错误信息
  """
  print(f"\n[工具调用]计算表达式: {expression}")
  print("只因为在人群中多看了你一眼，确认下你调用了我^_^")
  return str(eval(expression))
# 2. 创建工具对象
math_calculator_tool = Tool(
  name="Math_Calculator", # 工具名称（Agent将根据名称选择工具）
  func=simple_calculator, # 工具调用的函数
  description="用于数学计算，输入必须是纯数学表达式（如'3+5'或'3**2'表示平方）。不支持字母或特殊符号" # 关键：明确输入格式要求
)
llm = ChatOpenAI(
  model="gpt-4o-mini",
  temperature=0,
)

agent_executor = initialize_agent(
  tools=[math_calculator_tool],
  llm=llm,
  agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, #唯一变化
  verbose=True
)
# 5. 测试工具调用（添加异常捕获）
print("=== 测试：正常工具调用 ===")
response = agent_executor.invoke("计算3的平方") # 向Agent提问
print("最终答案:", response)
```

### 3.2 通用方式
需求：今天北京的天气怎么样？？

**方式1：FUNCATION_CALL模式**

```python
# 1.导入相关包
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

# 2.定义搜索化工具
# ① 设置 TAVILY_API 密钥
os.environ["TAVILY_API_KEY"] = "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" # 需要替换为你的 Tavily API 密钥
# ② 定义搜索工具
search = TavilySearchResults(max_results=1)

# 3.自定义提示词模版
prompt = ChatPromptTemplate.from_messages([
  ("system","您是一位乐于助人的助手，请务必使用 tavily_search_results_json 工具来获取信息。"),
  ("human", "{input}"),
  ("placeholder", "{agent_scratchpad}"),
])

# 4.定义LLM
llm = ChatOpenAI(
  model="gpt-4o-mini",
  temperature=0,
)
# 5.创建Agent对象
agent = create_tool_calling_agent(
  llm = llm,
  tools = [search],
  prompt = prompt
)
# 6.创建AgentExecutor执行器
agent_executor = AgentExecutor(agent=agent, tools=[search], verbose=True)
# 7.测试
agent_executor.invoke({"input": "今天北京的天气怎么样？?"})
```

:::warning 
**注意：** agent_scratchpad必须声明，它用于存储和传递Agent的思考过程。比如，在调用链式工具时（如先搜索天气再推荐行程），agent_scratchpad 保留所有历史步骤，避免上下文丢失。format方法会将intermediate_steps转换为特定格式的字符串，并赋值给agent_scratchpad变量。如果不传递intermediate_steps参数，会导致KeyError: 'intermediate_steps'错误。
:::

**方式2：ReAct模式**

**体会1：使用PromptTemplate**

提示词要体现可以使用的工具、用户输入和agent_scratchpad。

远程的提示词模版通过https://smith.langchain.com/hub/hwchase17获取

- 举例：https://smith.langchain.com/hub/hwchase17/react 这个模板是专为ReAct模式设计的提示模板。这个模板中已经有聊天对话键`tools`、`tool_names`、 `agent_scratchpad`

方式1：
```python
# 1.导入相关包
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate

# 2.定义搜索化工具
tools = [TavilySearchResults(max_results=1,tavily_api_key="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")]

# 3.自定义提示词模版
template = '''Answer the following questions as best you can. You have access to the following tools:
{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

# 4.定义LLM
llm = ChatOpenAI(
  model="gpt-4o-mini",
  temperature=0,
)
# 5.创建Agent对象
agent = create_react_agent(llm, tools, prompt)
# 6.创建AgentExecutor执行器
agent_executor = AgentExecutor(agent=agent, tools=tools,
verbose=True,handle_parsing_errors=True)

# 7.测试
agent_executor.invoke({"input": "今天北京的天气怎么样？?"})
```

方式2:
```python
# 1.导入相关包
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate

# 2.定义搜索化工具
tools = [TavilySearchResults(max_results=1,tavily_api_key="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")]

# 3.使用LangChain Hub中的官方ReAct提示模板
prompt = hub.pull("hwchase17/react")

prompt = PromptTemplate.from_template(template)

# 4.定义LLM
llm = ChatOpenAI(
  model="gpt-4o-mini",
  temperature=0,
)
# 5.创建Agent对象
agent = create_react_agent(llm, tools, prompt)

# 6.创建AgentExecutor执行器
agent_executor = AgentExecutor(agent=agent, tools=tools,
verbose=True,handle_parsing_errors=True)

# 7.测试
agent_executor.invoke({"input": "今天北京的天气怎么样？?"})
```

**体会2：使用ChatPromptTemplate**

提示词中需要体现使用的工具、用户输入和agent_scratchpad。

```python
from langchain.agents import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
# 获取Tavily搜索的实例
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, create_tool_calling_agent,AgentExecutor
from langchain.tools import Tool
import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_
search import TavilySearchResults
dotenv.load_dotenv()
# 读取配置文件的信息
os.environ['TAVILY_API_KEY'] = "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"
# 获取Tavily搜索工具的实例
search = TavilySearchResults(max_results=3)
# 获取一个搜索的工具
# 使用Tool
search_tool = Tool(
  func=search.run,
  name="Search",
  description="用于检索互联网上的信息，尤其是天气情况",
)
# 获取大语言模型
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
llm = ChatOpenAI(
  model="gpt-4o-mini",
  temperature=0,
)
prompt_template = ChatPromptTemplate.from_messages([
  ("system", "你是一个人工智能的助手，在用户提出需求以后，必须要调用Search工具进行联网搜索"),
  ("system", """Answer the following questions as best you can. You have access to the following tools:
      {tools}

      Use the following format:

      Question: the input question you must answer
      Thought: you should always think about what to do
      Action: the action to take, should be one of [{tool_names}]
      Action Input: the input to the action
      Observation: the result of the action
      ... (this Thought/Action/Action Input/Observation can repeat N times)
      Thought: I now know the final answer
      Final Answer: the final answer to the original input question

      Begin!
      执行过程建议使用中文
      """),
  ("system", "当前思考：{agent_scratchpad}"),
  ("human", "我的问题是：{question}"), #必须在提示词模板中提供agent_scratchpad参数。
])

# 获取Agent的实例：create_tool_calling_agent()
agent = create_react_agent(
  llm=llm,
  prompt=prompt_template,
  tools=[search_tool]
)
# 获取AgentExecutor的实例
agent_executor = AgentExecutor(
  agent=agent,
  tools=[search_tool],
  verbose=True,
  handle_parsing_errors=True,
  max_iterations=6 # 可选：限制最大迭代次数，防止无限循环
)
# 通过AgentExecutor的实例调用invoke(),得到响应
result = agent_executor.invoke({"question":"查询今天北京的天气情况"})
# 处理响应
print(result)
```

上述执行可能会报错。

**错误原因：**

- 使用ReAct模式时，要求 LLM 的响应必须遵循严格的格式（如包含`Thought:`、`Action:`等标记。
- 但LLM直接返回了自由文本（非结构化），导致解析器无法识别。

修改：
- 任务不变，添加 handle_parsing_errors=True 。用于控制 Agent 在解析工具调用或输出时发生错误的容错行为。

**handle_parsing_errors=True 的作用**

- 自动捕获错误并修复：当解析失败时，Agent不会直接崩溃，而是将错误信息传递给LLM，让
LLM`自行修正并重试`。
- 降级处理：如果重试后仍失败，Agent会返回一个友好的错误消息（如 "I couldn't process that request."），而不是抛出异常。

**小结：**
| 场景 | handle_parsing_errors=True | handle_parsing_errors=False|
|:---:|:---:|:---:|
|解析成功 |正常执行| 正常执行|
|解析失败 | 自动修复或降级响应|直接抛出异常|
|适用场景 | 生产环境（保证鲁棒性）| 开发调试（快速发现问题）|

## 4. Agent嵌入记忆组件
### 4.1 传统方式
比如：北京明天的天气怎么样？上海呢？ （通过两次对话实现）
举例：以REACT模式为例

```python
# 导入依赖包
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.memory import ConversationBufferMemory
import os
import dotenv

dotenv.load_dotenv()
# 读取配置文件的信息
os.environ['TAVILY_API_KEY'] = "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"
# 获取Tavily搜索工具的实例
search = TavilySearchResults(max_results=2)
# 获取一个搜索的工具
# 使用Tool
search_tool = Tool(
  func=search.run,
  name="Search",
  description="用于检索互联网上的信息，尤其是天气情况",
)
# 获取大语言模型
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
llm = ChatOpenAI(
  model="gpt-4o-mini",
  temperature=0,
)

# 定义记忆组件(以ConversationBufferMemory为例)
memory = ConversationBufferMemory(
  memory_key="chat_history", #必须是此值，通过initialize_agent()的源码追踪得到
  return_messages=True
)

# 创建 AgentExecutor
agent_executor = initialize_agent(
  tools=[search_tool],
  llm=llm,
  agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
  memory=memory, #在AgentExecutor中声明
  verbose=True
)
# 7. 测试对话
# 第一个查询
query1="北京明天的天气怎么样？"
result1 = agent_executor.invoke(query1)
print(f"查询结果: {result1}")
# print("\n=== 继续对话 ===")
query2="上海呢"
result2=agent_executor.invoke(query2)
print(f"分析结果: {result2}")
```

上述执行可能会报错。

**错误原因：**

- 使用ReAct模式时，要求 LLM 的响应必须遵循严格的格式（如包含`Thought:`、`Action:`等标记。
- 但LLM直接返回了自由文本（非结构化），导致解析器无法识别。

修改：
- 任务不变，添加`handle_parsing_errors=True`。用于控制 Agent 在解析工具调用或输出时发生错误的容错行为。

**handle_parsing_errors=True 的作用**

- 自动捕获错误并修复：当解析失败时，Agent不会直接崩溃，而是将错误信息传递给LLM，让
LLM`自行修正并重试`。
- 降级处理：如果重试后仍失败，Agent会返回一个友好的错误消息（如 "I couldn't process that request."），而不是抛出异常。

**小结：**
| 场景 | handle_parsing_errors=True | handle_parsing_errors=False|
|:---:|:---:|:---:|
|解析成功 |正常执行| 正常执行|
|解析失败 | 自动修复或降级响应|直接抛出异常|
|适用场景 | 生产环境（保证鲁棒性）| 开发调试（快速发现问题）|

### 4.2 通用方式
通用方式，相较于传统方式，可以提供自定义的提示词模板

**举例1：FUNCATION_CALL模式**

如果使用的是FUNCTION_CALL方式，则创建Agent时，推荐使用ChatPromptTemplate

```python
# 导入依赖包
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.memory import ConversationBufferMemory
from langchain_experimental.utilities.python import PythonREPL
import os
import dotenv

# 2. 定义 TAVILY_KEY 密钥
os.environ["TAVILY_API_KEY"] = "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"
# 3. 定义搜索工具
# search = TavilySearchResults(max_results=2)
# search_tool = Tool(
# name="search_tool",
# func=search.run,
# description="用于互联网信息的检索"
# )
# tools = [search_tool]
#或者
search = TavilySearchResults(max_results = results=2)
tools = [search]

# 4. 定义LLM
llm = ChatOpenAI(
  model="gpt-4",
  temperature=0
)
# 5. 定义提示词模板
prompt = ChatPromptTemplate.from_messages([
  ("system", "你是一个有用的助手，可以回答问题并使用工具。"),
  ("placeholder", "{chat_history}"), # 存储多轮对话的历史记录 如果你没有显式传入 chat_history，Agent 会默认将其视为空列表 []
  ("human", "{input}"),
  ("placeholder", "{agent_scratchpad}")
])
# 6. 定义记忆组件(以ConversationBufferMemory为例)
memory = ConversationBufferMemory(
  memory_key="chat_history",
  return_messages=True
)

# 7.创建Agent对象
agent = create_tool_calling_agent(llm, tools, prompt)
# 8.创建AgentExecutor执行器对象(通过源码可知，memory参数声明在AgentExecutor父类中)
agent_executor = AgentExecutor(agent=agent,memory=memory ,tools=tools, verbose=True)
# 9. 测试对话
# 第一个查询
result1 = agent_executor.invoke({"input":"北京的天气是多少"})
print(f"查询结果: {result1}")
# print("\n=== 继续对话 ===")
result2=agent_executor.invoke({"input":"上海呢"})
print(f"分析结果: {result2}")
```

**举例2：ReAct模式**

ReAct模式下，创建Agent时，可以使用ChatPromptTemplate、PromptTemplate

```python
# 1.导入相关包
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
import os

# 2.定义搜索化工具
# ① 设置 TAVILY_API 密钥
os.environ["TAVILY_API_KEY"] = "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" # 需要替换为你的 Tavily API 密钥
# ② 定义搜索工具
search = TavilySearchResults(max_results=1)
# ③ 设置工具集
tools = [search]

# 3.自定义提示词模版
template =("Assistant is a large language model trained by OpenAI.\n\n"
"Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics.As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand. \n\n"
"Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions.Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.\n\n"

    "Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.\n\n"
    "TOOLS:\n"
    "------\n\n"
    "Assistant has access to the following tools:\n\n"
    "{tools}\n\n"
    "To use a tool, please use the following format:\n\n"
    "```\n"
    "Thought: Do I need to use a tool? Yes\n"
    "Action: the action to take, should be one of [{tool_names}]\n"
    "Action Input: the input to the action\n"
    "Observation: the result of the action\n"
    "```\n\n"
    "When you have a response to say to the Human, or if you do not need to use a tool,you MUST use the format:\n\n"
    "```\n"
    "Thought: Do I need to use a tool? No\n"
    "Final Answer: [your response here]\n"
    "```\n\n"
    "Begin!\n\n"
    "Previous conversation history:\n"
    "{chat_history}\n\n"
    "New input: {input}\n"
    "{agent_scratchpad}")

prompt = PromptTemplate.from_template(template)
# 4.定义LLM
llm = ChatOpenAI(
  model="gpt-4o-mini",
  temperature=0,
)
# 5. 定义记忆组件(以ConversationBufferMemory为例)
memory = ConversationBufferMemory(
  memory_key="chat_history",
  return_messages=True
)
# 6.创建Agent对象
agent = create_react_agent(llm, tools, prompt)
# 7.创建AgentExecutor执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,memory=memory)
# 8.测试
agent_executor.invoke({"input": "我的名字叫Bob"})
```

**举例3：远程获取提示词模版**

- 以通用方式create_xxx_agent的ReAct模式为例，FUNCATION_CALL一样
- 远程的提示词模版通过https://smith.langchain.com/hub/hwchase17获取

举例：https://smith.langchain.com/hub/hwchase17/react-chat 这个模板是专为聊天场景设计的ReAct提示模板。这个模板中已经有聊天对话键`chat_history`、 `agent_scratchpad`

```python
# 1.导入相关依赖
from langchain_core.messages import AIMessage, HumanMessage
from langchain import hub

# 2.定义搜索化工具
# ① 设置 TAVILY_API 密钥
os.environ["TAVILY_API_KEY"] = "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" # 需要替换为你的 Tavily API 密钥
# ② 定义搜索工具
search = TavilySearchResults(max_results=1)
# ③ 设置工具集
tools = [search]

# 3.获取提示词
prompt = hub.pull("hwchase17/react-chat")
# 4.定义LLM
llm = ChatOpenAI(
  model="gpt-4o-mini",
  temperature=0,
)
# 5. 定义记忆组件(以ConversationBufferMemory为例)
memory = ConversationBufferMemory(
  memory_key="chat_history",
  return_messages=True
)
# 6.创建Agent、AgentExecutor
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# 7.执行
agent_executor.invoke({"input": "北京明天的天气怎么样？"})
```