# LangChain Retrieval
Retrieval直接翻译过来即“检索”，本章Retrieval模块包括与检索步骤相关的所有内容，例如数据的获取、切分、向量化、向量存储、向量检索等模块。常被应用于构建一个“ 企业/私人的知识库 ”，提升大模型的整体能力。

## 1. Retrieval模块的设计意义
### 1.1 大模型的幻觉问题
拥有记忆后，确实扩展了AI工程的应用场景。
但是在专有领域，LLM无法学习到所有的专业知识细节，因此在 面向专业领域知识 的提问时，无法给出可靠准确的回答，甚至会“胡言乱语”，这种现象称之为`LLM的“幻觉”`。
大模型生成内容的不可控，尤其是在金融和医疗领域等领域，**一次金额评估的错误，一次医疗诊断的失误，哪怕只出现一次都是致命的**。但对于非专业人士来说可能难以辨识。目前还没有能够百分之百解决这种情况的方案。

**当前大家普遍达成共识的一个方案：**
首先，为大模型提供一定的上下文信息，让其输出会变得更稳定。
其次，利用本章的RAG，将检索出来的 文档和提示词 输送给大模型，生成更可靠的答案。

### 1.2 RAG的解决方案
可以说，当应用需求集中在利用大模型去**回答特定私有领域的知识**，且知识库足够大，那么除了`微调大模型`外，**RAG**就是非常有效的一种缓解大模型推理的“幻觉”问题的解决方案。
LangChain对这一流程提供了解决方案。

> 如果说LangChain相当于给LLM这个“大脑”安装了“四肢和躯⼲”，RAG则是为LLM提供了接⼊“
⼈类知识图书馆”的能⼒。

### 1.3 RAG的优缺点
**RAG的优点**

1）相比提示词工程，RAG有`更丰富的上下文和数据样本`，可以不需要用户提供过多的背景描述，就能生成比较符合用户预期的答案。
2）相比于模型微调，RAG可以提升问答内容的 `时效性` 和 `可靠性`。
3）在一定程度上保护了业务数据的 `隐私性。

**RAG的缺点**

1）由于每次问答都涉及外部系统数据检索，因此RAG的 `响应时延` 相对较高。
2）引用的外部知识数据会 `消耗大量的模型Token` 资源。

### 1.4 Retrieval流程

![alt text](/public/langchain/retrieval/1.png)

**环节1:Source(数据源)**
指的是RAG架构中所外挂的知识库。这里有三点说明：
1、原始数据源类型多样：如：视频、图片、文本、代码、文档等
2、形式多样性：
- 可以是上百个.csv文件，可以是千万个.json文件，也可以是上万个.pdf文件
- 可以是某一个业务流程外放的API，可以是某个网站的实时数据等

**环节2:Load(加载)**

文档加载器（Document Loaders）负责将来自不同数据源的非结构化文本，加载到内存，成为文档（Document）对象
文档对象包含`文档内容`和相关`元数据信息`，例如TXT、CSV、HTML、JSON、Markdown、PDF，甚至YouTube视频转录等。

文档加载器还支持“**延迟加载**”模式，以缓解处理大文件时的内存压力。

**环节3：Transform（转换）**
**文档转换器(Document Transformers)** 负责对加载的文档进行转换和处理，以便更好地适应下游任务的需求。
文档转换器提供了一致的接口（工具）来操作文档，主要包括以下几类：
- 文本拆分器(Text Splitters) ：将长文本拆分成语义上相关的小块，以适应语言模型的上下文窗口限制。
- 冗余过滤器(Redundancy Filters) ：识别并过滤重复的文档。
- 元数据提取器(Metadata Extractors) ：从文档中提取标题、语调等结构化元数据。
- 多语言转换器(Multi-lingual Transformers) ：实现文档的机器翻译。
- 对话转换器(Conversational Transformers) ：将非结构化对话转换为问答格式的文档。
总的来说，文档转换器是 LangChain 处理管道中非常重要的一个组件，它丰富了框架对文档的表示和操作能力。
在这些功能中，文档拆分器是必须的操作。下面单独说明。

**环节3.1：Text Splitting（文档拆分）**
- 拆分/分块的必要性 ：前一个环节加载后的文档对象可以直接传入文档拆分器进行拆分，而文档切块后才能 向量化 并存入数据库中。
- 文档拆分器的多样性 ：LangChain提供了丰富的文档拆分器，不仅能够切分普通文本，还能切分Markdown、JSON、HTML、代码等特殊格式的文本。
- 拆分/分块的挑战性 ：实际拆分操作中需要处理许多细节问题，`不同类型的文本`、 `不同的使用场景`都需要采用不同的分块策略。
    - 可以按照 `数据类型` 进行切片处理，比如针对 文本类数据 ，可以直接按照字符、段落进行切片； `代码类数据` 则需要进一步细分以保证代码的功能性；
    - 可以直接根据 `token` 进行切片处理

**在构建RAG应用程序的整个流程中，拆分/分块是最具挑战性的环节之一，它显著影响检索效果。**目前还没有通用的方法可以明确指出哪一种分块策略最为有效。不同的使用场景和数据类型都会影响分块策略的选择。

**环节4：Embed（嵌入）**

文档嵌入模型（Text Embedding Models）负责将 文本 转换为 向量表示 ，即模型赋予了文本计算机可
理解的数值表示，使文本可用于向量空间中的各种运算，大大拓展了文本分析的可能性，是自然语言处理领域非常重要的技术。

![alt text](/public/langchain/retrieval/2.png)

文本嵌入为 LangChain 中的问答、检索、推荐等功能提供了重要支持。具体为：
- 语义匹配 ：通过计算两个文本的向量余弦相似度，判断它们在语义上的相似程度，实现语义匹配。
- 文本检索 ：通过计算不同文本之间的向量相似度，可以实现语义搜索，找到向量空间中最相似的文本。
- 信息推荐 ：根据用户的历史记录或兴趣嵌入生成用户向量，计算不同信息的向量与用户向量的相似度，推荐相似的信息。
- 知识挖掘 ：可以通过聚类、降维等手段分析文本向量的分布，发现文本之间的潜在关联，挖掘知识。
- 自然语言处理 ：将词语、句子等表示为稠密向量，为神经网络等下游任务提供输入。

**环节5：Store（存储）**
LangChain 还支持把文本嵌入存储到向量存储或临时缓存，以避免需要重新计算它们。这里就出现了数据库，支持这些嵌入的高效 `存储` 和 `搜索` 的需求。

![alt text](/public/langchain/retrieval/3.png)

**环节6：Retrieve（检索）**
检索器（Retrievers）是一种用于`响应非结构化查询`的接口，它可以返回符合查询要求的文档。
LangChain 提供了一些常用的检索器，如`向量检索器`、`文档检索器` 、`网站研究检索器`等。
通过配置不同的检索器，LangChain可以灵活地平衡检索的精度、召回率与效率。检索结果将为后续的问答生成提供信息支持，以产生更加准确和完整的回答。

## 2. 文档加载器 Document Loaders
LangChain的设计：对于 Source 中多种不同的数据源，我们可以用一种统一的形式读取、调用。
### 2.1 加载txt

```python
# 1.导入相关依赖
from langchain.document_loaders import TextLoader
# 2.定义TextLoader对象，file_path=".txt的位置"
text_loader = TextLoader(file_path="asset/load/01-langchain-utf-8.txt", encoding="utf-8")
# 3.加载
docs = text_loader.load() #返回List列表(Document对象)
# 4.打印
print(docs)
```
```
[Document(metadata={'source':'asset/load/01-langchain-utf-8.txt'},page_content='LangChain'是⼀个⽤于构建基于⼤语⾔模型（LLM）应⽤的开发框架，旨在帮助开发者更⾼效地集成、管理和增强⼤语⾔模型的能⼒，构建端到端的应⽤程序。它提供了⼀套模块化⼯具和接口，⽀持从简单的⽂本⽣成到复杂的多步骤推理任务')]
```

Documment对象中有两个重要的属性：
- page_content：真正的文档内容
- metadata：文档内容的原数据

```python
type(docs[0]) #langchain_core.documents.base.Document
docs[0].page_content
'''
LangChain 是一个用于开发由大型语言模型 (LLMs) 驱动的应用程序的框架。LangChain简化了LLM应用程序生命周期的每个阶段。\nLangChain 已经成为了我们每一个大模型开发工程师的标配。
'''
docs[0].metadata # {'source': './data/langchain.txt'}
```

### 2.2 加载pdf
**举例1：**
LangChain加载PDF文件使用的是pypdf，先安装
> pip install pypdf

```python
# 1.导入相关的依赖 PyPDFLoader()
from langchain.document_loaders import PyPDFLoader
# 2.定义PyPDFLoader
pdfLoader = PyPDFLoader(file_path="asset/load/02-load.pdf")
# 3.加载
docs = pdfLoader.load()
print(docs)
print(type(docs[0]))
# # 4.遍历集合
# for doc in docs:
# print(f"加载的文档:{doc.page_content}")
```

### 2.3 加载CSV
**举例1** ：加载csv所有列

```python
from langchain_community.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(
    file_path="./asset/load/03-load.csv",
    source_column='author' # 使用 source_column 参数指定文件加载的列，保存在source变量中。
)
data = loader.load()
print(data)
print(type(data)) # <class 'list'>
print(type(data[0])) # <class 'langchain_core.documents.base.Document'>
print(len(data)) # 4
print(data[0].page_content) # id: 1 title: Introduction to Python ...
```

### 2.4 加载JSON
LangChain提供的JSON格式的文档加载器是 **JSONLoader** 。在实际应用场景中，JSON格式的数据占有很大比例，而且JSON的形式也是多样的。我们需要特别关注。
JSONLoader 使用指定的 jq结构来解析 JSON 文件。jq是一个轻量级的命令行 JSON 处理器 ，可以对JSON 格式的数据进行各种复杂的处理，包括数据过滤、映射、减少和转换，是处理 JSON 数据的首选工具之一。
> pip install jq

**举例1** ：使用JSONLoader文档加载器加载

```python
# 1.导入依赖
from langchain_community.document_loaders import JSONLoader
from pprint import pprint
# 2.定义JSONLoader对象
# 错误的
# json_loader=JSONLoader(file_path="asset/load/04-load.json")
# 情况1
# json_loader=JSONLoader(
# file_path="asset/load/04-load.json",
# jq_schema=".", #直接提取完整的JSON对象（包括所有字段）
# text_content=False #保持原始 JSON 结构，将提取的数据转换为JSON字符字段中
# )串存入page_content

# 情况2
# .messages[].content:遍历.messages[]中所有元素 从每一个元素中提取.content字段
json_loader=JSONLoader(
    file_path="asset/load/04-load.json",
    jq_schema=".messages[].content"
)
# 3.加载
docs = json_pprint(docs)
loader.load()
# 4.提取content中指定字符数的内容
# print(docs[0].page_content[:10])
```

**举例2** ：提取04-response.json文件中嵌套在 data.items[].content 的文本
- 如果希望处理 JSON 中的 **嵌套字段、数组元素提取**，可以使用 content_key 配合is_content_key_jq_parsable=True ，通过 jq 语法精准定位目标数据。
- 通常，对api请求结果的采集

```python
# 1.导入相关依赖
from langchain_community.document_loaders import JSONLoader
from pprint import pprint
# 2.定义json文件的路径
file_path = 'asset/load/04-response.json'
# 3.定义JSONLoader对象
# 提取嵌套在 data.items[].content 的文本，并保留其他字段作为元数据
# 方式1：
# loader = JSONLoader(
# file_path=file_path,
# jq_schema=".data.items[].content",
# )
# 方式2：
loader = JSONLoader(
file_path=file_path,
jq_schema=".data.items[]", # 先定位到数组条目
content_key=".content", # 再从条目中提取 content 字段
is_content_key_jq_parsable=True # 用jq解析content_key
)
# 4.加载
data = loader.load()
pprint(data)
pprint(data[0].page_content)
```

### 2.5 加载HTML（了解）
> pip install unstructured

```python
# 1.导入相关的依赖
from langchain.document_loaders import UnstructuredHTMLLoader
# 2.定义UnstructuredHTMLLoader对象
# strategy:
# "fast" 解析加载html文件速度是比较快（但可能丢失部分结构或元数据）
# "hi_res": (高分辨率解析) 解析精准（速度慢一些）
# "ocr_only" 强制使用ocr提取文本，仅仅适用于图像（对HTML无效）
# mode ：one of `{'paged', 'elements', 'single'}
# "elements" 按语义元素（标题、段落、列表、表格等）拆分成多个独立的小文档
html_loader = UnstructuredHTMLLoader(
    file_path="asset/load/05-load.html",
    mode="elements",
    strategy="fast"
)
# 3.加载
docs = html_loader.load()
print(len(docs)) # 16
# 4.打印
for doc in docs:
    print(doc)
```

### 2.6 加载Markdown(了解)
将Markdown文档按语义元素（标题、段落、列表、表格等）拆分成多个独立的小文档（ Element 对象），而不是返回单个大文档。通过指定 `mode="elements"` 轻松保持这种分离。每个分割后的元素会包含元数据。

```python
# 1.导入相关的依赖
from langchain.document_loaders import UnstructuredMarkdownLoader
from pprint import pprint
# 2.定义UnstructuredMarkdownLoader对象
md_loader = UnstructuredMarkdownLoader(
    file_path="./asset/load/06-load.md",
    mode= "elements",
    strategy="fast"
)
# 3.加载
docs = md_loader.load()
print(len(docs))
# 4.打印
for doc in docs:
# pprint(doc)
pprint(doc.page_content)
```

### 2.7 加载File Directory(了解)
除了上述的单个文件加载，我们也可以批量加载一个文件夹内的所有文件。
> pip install unstructured

```python
# 1.导入相关的依赖
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PythonLoader
from pprint import pprint
# 2.定义DirectoryLoader对象,指定要加载的文件夹路径、要加载的文件类型和是否使用多线程
directory_loader = DirectoryLoader(
  path="./asset/load",
  glob="*.py",
  use_multithreading=True,
  show_progress=True,
  loader_cls=PythonLoader
)
# 3.加载
docs = directory_loader.load()
# 4.打印
print(len(docs))
for doc in docs:
  pprint(doc)
```

### 2.8 BaseLoader、Document源码分析
一方面：LangChain在设计时，要保证Source中多种不同的数据源，在接下来的流程中可以用一种统一的形式读取、调用。
另一方面：为什么 `PDFloader` 和 `TextLoader` 等Document Loader 都使用 load() 去加载，且都使
用 `.page_content` 和 `.metadata` 读取数据。
【解答】每一个在LangChain中集成的文档加载器，都要继承自 **BaseLoader(文档加载器)** ，BaseLoader提供了一个名为"load"的公开方法，用于从配置的不同 数据源 加载数据，全部作为`Document`对象。实现逻辑如下所示：
**BaseLoader类分析**
BaseLoader类定义了如何从不同的数据源加载文档，每个基于不同数据源实现的loader，都需要继承**BaseLoader**。Baseloader要求不多，对于任何具体实现的loader，最少都要实现load方法。

```python
class BaseLoader(ABC):
"""文档加载器接口。
实现应当使用生成器实现延迟加载方法，以避免一次性将所有文档加载进内存。
`load` 方法仅供用户方便使用，不应被重写。
"""
# 子类不应直接实现此方法。而应实现延迟加载方法。
def load(self) -> List[Document]:
"""将数据加载为 Document 对象。"""
return list(self.lazy_load())
def load_and_split(
  self, text_splitter: Optional[TextSplitter] = None
) -> List[Document]:
"""加载文档并将其分割成块。块以 Document 形式返回。
不要重写此方法。它应被视为已弃用！
参数:
  text_splitter: 用于分割文档的 TextSplitter 实例。默认为 RecursiveCharacterTextSplitter。
返回:
  文档列表。
"""
.....
.....
  _text_splitter: TextSplitter = RecursiveCharacterTextSplitter()
else:
  _text_splitter = text_splitter
  docs = self.load()
  return _text_splitter.split_documents(docs)
```
`BaseLoader` 把数据加载成 **Documents object** ，存到 `Documents` 类中的 `page_content` 中。

**Document类分析**
`Document` 允许用户与文档的内容进行交互，可以查看文档内容。

```python
class Document(Serializable):
  ""用于存储文本及其关联元数据的类。"""
page_content: str
"""字符串文本。"""
metadata: dict = Field(default_factory=dict)
"""关于页面内容的任意元数据（例如，来源、与其他文档的关系等）。"""
type: Literal["Document"] = "Document"

def __init__(self, page_content: str,**kwargs: Any) -> None:
  """将 page_content 作为位置参数或命名参数传入。"""
  super().__init__(page_content=page_content,**kwargs)

@classmethod
def is_lc_serializable(cls) -> bool:
  """返回此类是否可序列化。"""
  return True

@classmethod
def get_lc_namespace(cls) -> List[str]:
  """获取 langchain 对象的命名空间。"""
  return ["langchain", "schema", "document"]
```
通过 存 + 读的两个基类的抽象，满足不同类型加载器在数据形式上的统一。除此之外，其中的
`metadata`会根据loader实现的不同写入不同的数据，同样是一个必要的基础属性。

## 3.文档拆分器 Text Splitters
### 3.1 为什么要拆分/分块/切分
当拿到统一的一个Document对象后，接下来需要切分成Chunks。如果不切分，而是考虑作为一个整体的Document对象，会存在两点问题：
> 1. 假设提问的Query的答案出现在某一个Document对象中，那么将检索到的整个Document对象直接放入Prompt中并 不是最优的选择 ，因为其中一定会 包含非常多无关的信息 ，而无效信息越多，对大模型后续的推理影响越大。
> 2. 任何一个大模型都存在最大输入的 Token限制 ，如果一个Document非常大，比如一个几百兆的PDF，那么大模型肯定无法容纳如此多的信息。

基于此，一个有效的解决方案就是将完整的Document对象进行**分块处理（Chunking)**。无论是在存储还是检索过程中，都将以这些**块(chunks)**为基本单位，这样有效地避免内容不相关性问题和超出最大输入限制的问题。

### 3.2 Chunking拆分的策略
方法1：根据句子切分：这种方法按照自然句子边界进行切分，以保持语义完整性。

方法2：按照固定字符数来切分：这种策略根据特定的字符数量来划分文本，但可能会在不适当的位置切断句子。

方法3：按固定字符数来切分，结合重叠窗口（overlapping windows）：此方法与按字符数切分相似，但通过重叠窗口技术避免切分关键内容，确保信息连贯性。

方法4：递归字符切分方法：通过递归字符方式动态确定切分点，这种方法可以根据文档的复杂性和内容密度来调整块的大小。

方法5：根据语义内容切分：这种 高级策略 依据文本的语义内容来划分块，旨在保持相关信息的集中和完整，适用于需要高度语义保持的应用场景。

> 第2种⽅法（按照字符数切分）和第3种⽅法（按固定字符数切分结合重叠窗口）主要基于字符进⾏⽂本的切分，而不考虑⽂章的实际内容和语义。这种⽅式虽简单，但可能会导致 `主题或语义上的断裂` 。
相对而⾔，第4种递归⽅法更加灵活和⾼效，它结合了固定⻓度切分和语义分析。通常是 `首选策`
略 ，因为它能够更好地确保每个段落包含⼀个完整的主题。
而第5种⽅法，基于语义的分割虽然能精确地切分出完整的主题段落，但这种⽅法效率较低。它需
要运⾏复杂的分段算法（segmentation algorithm）， `处理速度较慢` ，并且 `段落长度可能极不均匀` （有的主题段落可能很⻓，而有的则较短）。因此，尽管它在某些需要⾼精度语义保持的场景下有其应⽤价值，但并 `不适合所有情况` 。

这些方法各有优势和局限，选择适当的分块策略取决于具体的应用需求和预期的检索效果。接下来我们依次尝试用常规手段应该如何实现上述几种方法的文本切分。

小结：几个常用的文档切分器的方法的调用

```python

#方式1：传入的参数类型：字符串; 返回值类型：List[str]
split_text(xxx)

#方式2：传入的参数类型：List[str]; 返回值类型：List[Document]
create_documents(xxx) #底层调用了split_text(xxx)

#方式3：传入的参数类型：List[Document]; 返回值类型：List[Document
split_documents(xxx) #底层调用了create_documents()
```
此外，这里提供了一个可视化展示文本如何分割的工具，https://chunkviz.up.railway.app/

### 3.3 具体实现
LangChain提供了许多不同类型的文档切分器
官网地址：https://python.langchain.com/api_reference/text_splitters/index.html
#### 3.3.1 CharacterTextSplitter：Split by character
参数情况说明：
- `chunk_size` ：每个切块的最大token数量，默认值为4000。
- `chunk_overlap` ：相邻两个切块之间的最大重叠token数量，默认值为200。
- `separator` ：分割使用的分隔符，默认值为"\n\n"。
- `length_function` ：用于计算切块长度的方法。默认赋值为父类TextSplitter的len函数。

**举例1：**
```python
# 1.导入相关依赖
from langchain.text_splitter import CharacterTextSplitter
# 2.定义要分割的文本
text = "这是一个示例文本啊。我们将使用CharacterTextSplitter将其分割成小块。分割基于字符数。"
# text = """
# LangChain 是一个用于开发由语言模型。驱动的应用程序的框架的。它提供了一套工具和抽象。使开发者能够更容易地构建复杂的应用程序。
# ""
# 3.定义分割器实例
text_splitter = CharacterTextSplitter(
  chunk_size=30, # 每个块的最大字符数
  chunk_overlap=5, # 块之间的重叠字符数
  separator="。", # 按句号分割
)
# 4.开始分割
chunks = text_splitter.split_text(text)
# 5.打印效果
for i,chunk in enumerate(chunks):
  print(f"块{i + 1}:长度：{len(chunk)}")
  print(chunk)
  print("-"*50)
```
**注意：无重叠**

**separator优先原则**：当设置了 `separator` （如"。"），分割器会首先尝试在分隔符处分割，然后再考虑chunk_size。这是为了避免在句子中间硬性切断。这种设计是为了：

> 1. 优先保持语义完整性（不切断句子）
> 2. 避免产生无意义的碎片（如半个单词/不完整句子）
> 3. 如果 chunk_size 比片段小，无法拆分片段，导致 overlap失效。
> 4. chunk_overlap仅在合并后的片段之间生效（如果 `chunk_size` 足够大）。如果没有合并的片段，则 overlap失效。见举例3。

**举例2：**

**注意：有重叠**

此时，文本“这是第二段内容。”的token正好就是8。

```python
# 1.导入相关依赖
from langchain.text_splitter import CharacterTextSplitter
# 2.定义要分割的文本
text = "这是第一段文本。这是第二段内容。最后一段结束。"
# 3.定义字符分割器
text_splitter = CharacterTextSplitter(
  separator="。",
  chunk_size=20,
  chunk_overlap=8,
  keep_
  separator=True #chunk中是否保留切割符
)
# 4.分割文本
chunks = text_splitter.split_text(text)
# 5.打印结果
for i,chunk in enumerate(chunks):
  print(f"块{i + 1}:长度：{len(chunk)}")
  print(chunk)
  print("-"*50)
```

**3.3.2 RecursiveCharacterTextSplitter：最常用**

文档切分器中较常用的是`RecursiveCharacterTextSplitter`(递归字符文本切分器) ，遇到`特定字符`时进行分割。默认情况下，它尝试进行切割的字符包括 ["\n\n", "\n", " ", ""]。

具体为：根据第一个字符进行切块，但如果任何切块太大，则会继续移动到下一个字符继续切块，以此类推。
此外，还可以考虑添加，。等分割字符。

**特点：**
- **保留上下文**：优先在自然语言边界（如段落、句子结尾）处分割，`减少信息碎片化`。
- **智能分段**：通过递归尝试多种分隔符，将文本分割为大小接近chunk_size的片段。
- **灵活适配**：适用于多种文本类型（代码、Markdown、普通文本等），是LangChain中最通用的文本拆分器。

此外，还可以指定的参数包括：
- **chunk_size** ：同TextSplitter（父类）。
- **chunk_overlap** ：同TextSplitter（父类）。
- **length_function** ：同TextSplitter（父类）。
- **add_start_index** ：同TextSplitter（父类）。

**举例1：** 使用split_text()方法演示

```python
# 1.导入相关依赖
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 2.定义RecursiveCharacterTextSplitter分割器对象
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=10,
  chunk_overlap=0,
  add_start_index=True,
)
# 3.定义拆分的内容
text="LangChain框架特性\n\n多模型集成(GPT/Claude)\n记忆管理功能\n链式调用设计。文档分析场景示例：需要处理PDF/Word等格式。"
# 4.拆分器分割
paragraphs = text_splitter.split_text(text)
for para in paragraphs:
  print(para)
  print('-------')
```

**举例2：** 使用create_documents()方法演示，传入字符串列表，返回Document对象列表
```python
# 1.导入相关依赖
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 2.定义RecursiveCharacterTextSplitter分割器对象
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=10,
  chunk_overlap=0,
  add_start_index=True,
)
# 3.定义拆分的内容
list=["LangChain框架特性\n\n多模型集成(GPT/Claude)\n记忆管理功能\n链式调用设计。文档分析场景示例：需要处理PDF/Word等格式。"]
# 4.拆分器分割
# create_documents()：形参是字符串列表，返回值是Document的列表
paragraphs = text_splitter.create_documents(list)
for para in paragraphs:
  print(para)
  print('-------')
```

**举例3：** 使用create_documents()方法演示，将本地文件内容加载成字符串，进行拆分

```python
# 1.导入相关依赖
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 2.打开.txt文件
with open("asset/load/08-ai.txt", encoding="utf-8") as f:
  state_of_the_union = f.read() #返回的是字符串

# 3.定义RecursiveCharacterTextSplitter（递归字符分割器）
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=100,
  chunk_overlap=20,
  #chunk_overlap=0,
  length_function=len
)
# 4.分割文本
texts = text_splitter.create_documents([state_of_the_union])
# 5.打印分割文本
for text in texts:
  print(f"🔥{text.page_content}")
```

**举例4:** 用split_documents()方法演示，利用PDFLoader加载文档，对文档的内容用递归切割器切割

```python
# 1.导入相关依赖
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# 2.定义PyPDFLoader加载器
loader = PyPDFLoader("./asset/load/02-load.pdf")
# 3.加载和切割文档对象
docs = loader.load() # 返回Document对象构成的list
# print(f"第0页：\n{docs[0]}")
# 4.定义切割器
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=200,
  #chunk_size=120,
  chunk_overlap=0,
  # chunk_overlap=100,
  length_function=len,
  add_start_index=True,
)
# 5.对pdf内容进行切割得到文
档对象
paragraphs = text_splitter.split_documents(docs)
#paragraphs = text_splitter.create_documents([text])
for para in paragraphs:
  print(para.page_content)
  print('-------')
```

**举例5：** 自定义分隔符

有些书写系统没有单词边界，例如中文、日文和泰文。使用默认分隔符列表["\n\n", "\n", " ", ""]分割文本可能导致单词错误的分割。为了保持单词在一起，你可以自定义分割字符，覆盖分隔符列表以包含额外的标点符号。

```python
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=200,
  chunk_overlap=20, # 增加重叠字符
  separators=["\n\n", "\n", "。", "！", "？", "……", "，", ""], # 添加中文标点
  length_function=len,
  keep_separator=True #保留句尾标点（如 ……），避免切割后丢失语气和逻辑
)
```
效果：算法优先在句号、省略号处切割，保持句子完整性。

#### 3.3.2 TokenTextSplitter/CharacterTextSplitter：Split by tokens
当我们将文本拆分为块时，除了字符以外，还可以： 按Token的数量分割 （而非字符或单词数），将长文本切分成多个小块。

**什么是Token？**

- 对模型而言，Token是文本的最小处理单位。例如：
  - 英文："hello" → 1个Token，"ChatGPT" → 2个Token（"Chat" + "GPT" ）。
  - 中文："人工智能" → 可能拆分为2-3个Token（取决于分词器）。

**为什么按Token分割？**

- 语言模型对输入长度的限制是基于Token数（如GPT-4的8k/32k Token上限），直接按字符或单词分割可能导致实际Token数超限。（确保每个文本块不超过模型的Token上限）
- 大语言模型(LLM)通常是以token的数量作为其计量(或收费)的依据，所以采用token分割也有助于我们在使用时更方便的控制成本。

**TokenTextSplitter 使用说明：**

- 核心依据：Token数量 + 自然边界。（TokenTextSplitter 严格按照 token 数量进行分割，但同时会优先在自然边界（如句尾）处切断，以尽量保证语义的完整性。）
- 优点：与LLM的Token计数逻辑一致，能尽量保持语义完整
- 缺点：对非英语或特定领域文本，Token化效果可能不佳
- 典型场景：需要精确控制Token数输入LLM的场景

**举例1：** 使用TokenTextSplitter

```python
# 1.导入相关依赖
from langchain_text_splitters import TokenTextSplitter
# 2.初始化 TokenTextSplitter
text_splitter = TokenTextSplitter(
  chunk_size=33, #最大 token 数为 32
  chunk_overlap=0, #重叠 token 数为 0
  encoding_name="cl100k_base", # 使用 OpenAI 的编码器,将文本转换为 token 序列
)
# 3.定义文本
text = "人工智能是一个强大的开发框架。它支持多种语言模型和工具链。人工智能是指通过计算机程序模拟人类智能的一门科学。自20世纪50年代诞生以来，人工智能经历了多次起伏。"

# 4.开始切割
texts = text_splitter.split_text(text)
# 打印分割结果
print(f"原始文本被分割成了 {len(texts)} 个块:")
for i, chunk in enumerate(texts):
  print(f"块{i+1}: 长度：{len(chunk)} 内容：{chunk}")
  print("-" * 50)
```

**为什么会出现这样的分割？**

1、**`第一块 (29字符)`** ：内容是一个完整的句子，以句号结尾。TokenTextSplitter识别到这是一个自然的语义边界，即使这里的 token 数量可能尚未达到 33，它也选择在此处切割，以保证第一块语义的完整性。

2、**`第二块 (32字符)`** ：内容包含了另一个完整句子 **`“人工智能是指...一门科学。”`** 以及下一句的开头 “自20世纪50” 。分割器在处理完第一个句子的 token 后，可能 token 数量已经接近 **`chunk_size`** ，于是在下一个自然边界（这里是句号）之后继续读取了少量 token（“自20世纪50”），直到非常接近 33token 的限制。

**注意**：“50” 之后被切断，是因为编码器很可能将“50”识别为一个独立的 token，而“年代”是另一个 token。为了保证 token 的完整性，它不会在“50”字符中间切断。

3、**`第三块 (19字符)`** ：是第二块中断内容的剩余部分，形成了一个较短的块。这是因为剩余内容本身的 token 数量就较少。

特别注意：**字符长度不等于 Token 数量。**

**举例2**：使用CharacterTextSplitter

```python
# 1.导入相关依赖
from langchain_text_splitters import CharacterTextSplitter
import tiktoken # 用于计算Token数量

# 2.定义通过Token切割器
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
  encoding_name="cl100k_base", # 使用 OpenAI 的编码器
  chunk_size=18,
  chunk_overlap=0,
  separator="。", # 指定中文句号为分隔符
  keep_separator=False, # chunk中是否保留分隔符
)
# 3.定义文本
text = "人工智能是一个强大的开发框架。它支持多种语言模型和工具链。今天天气很好，想出去踏青。但是又比较懒不想出去，怎么办"

# 4.开始切割
texts = text_splitter.split_text(text)
print(f"分割后的块数: {len(texts)}")
# 5.初始化tiktoken编码器（用于Token计数）
encoder = tiktoken.get_encoding("cl100k_base") # 确保CharacterTextSplitter的encoding_name一致
# 6.打印每个块的Token数和内容
for i, chunk in enumerate(texts):
  tokens = encoder.encode(chunk) # 现在encoder已定义
  print(f"块{i + 1}: {len(tokens)} Token\n内容: {chunk}\n")
```

#### 3.3.3 SemanticChunker：语义分块
SemanticChunking（语义分块）是 LangChain 中一种更高级的文本分割方法，它超越了传统的基于字符或固定大小的分块方式，而是根据 文本的语义结构 进行智能分块，使每个分块保持 语义完整性 ，从而提高检索增强生成(RAG)等应用的效果。

**语义分割 vs 传统分割**

|特性| 语义分割（SemanticChunker）| 传统字符分割（RecursiveCharacter）|
|:---:|:---:|:---:|
|**分割依据** |嵌入向量相似度| 固定字符/换行符|
|**语义完整性** | ✅ 保持主题连贯 | ❌ 可能切断句子逻辑 |
|**计算成本** | ❌ 高（需嵌入模型）| ✅ 低 |
|**适用场景** | 需要高语义一致性的任务 | 简单文本预处理 |

**举例：**

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
import os
import dotenv

dotenv.load_dotenv()
# 加载文本
with open("asset/load/09-ai1.txt", encoding="utf-8") as f:
  state_of_the_union = f.read() #返回字符串

# 获取嵌入模型
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
embed_model = OpenAIEmbeddings(
  model="text-embedding-3-large"
)
# 获取切割器
text_splitter = SemanticChunker(
  embeddings=embed_model,
  breakpoint_threshold_type="percentile",#断点阈值类型：字面值["百分位数", "标差", "四分位准距", "梯度"] 选其一
  breakpoint_threshold_amount=65.0 #断点阈值数量 (极低阈值 → 高分割敏感度)
)
# 切分文档
docs = text_splitter.create_documents(texts = [state_of_the_union])
print(len(docs))
for doc in docs:
  print(f"🔍 文档 {doc}:")
```

**关于参数的说明：**

> 1. breakpoint_threshold_type （断点阈值类型）
- **作用**：定义文本语义边界的检测算法，决定何时分割文本块。
- 可选值及原理：

|类型 |原理说明| 适用场景|
|:---:|:----:|:----:|
|**`percentile`**|计算相邻句子嵌入向量的余弦距离，取**距离分布的第N百分位值**作为阈值，高于此值则分割|常规文本（如文章、报告）|
|**`standard_deviation`**|以**均值 + N倍标准差**为阈值，识别语义突变点|语义变化剧烈的文档（如技术手册）|
|**`interquartile`**| 用**四分位距（IQR）** 定义异常值边界，超过则分割|长文档（如书籍）|
|**`gradient`**|基于**嵌入向量变化的梯度**检测分割点（需自定义实现）| 实验性需求 |

> 2. breakpoint_threshold_amount （断点阈值量）

- **作用**：控制分割的**粒度敏感度**，值越小分割越细（块越多），值越大分割越粗（块越少）。
- 取值范围与示例：
  - **`percentile`** 模式：0.0~100.0，用户代码设 65.0 表示仅当余弦距离 > 所有距离中最低的65.0%值时分割 。默认值是：95.0，兼顾语义完整性与检索效率。值过小（比如0.1），会产生大量小文本块，过度分割可能导致上下文断裂。
  - **`standard_deviation`** 模式：浮点数（如 **1.5** 表示均值+1.5倍标准差）。
  - **`interquartile`** 模式：倍数（如 **1.5** 是IQR标准值）。

#### 3.3.4 其它拆分器
**类型1：HTMLHeaderTextSplitter：Split by HTML header**

HTMLHeaderTextSplitter是一种专门用于处理HTML文档的文本分割方法，它根据HTML的 标题标签（如\<h1\>、\<h2\>等） 将文档划分为逻辑分块，同时保留标题的层级结构信息。

**举例：**
```python
# 1.导入相关依赖
from langchain.text_splitter import HTMLHeaderTextSplitter
# 2.定义HTML文件
html_string = """
<!DOCTYPE html>
<html>
<body>
  <div>
    <h1>欢迎来到尚硅谷！</h1>
    <p>尚硅谷是专门培训IT技术方向</p>
    <div>
      <h2>尚硅谷老师简介</h2>
      <p>尚硅谷老师拥有多年教学经验，都是从一线互联网下来</p>
      <h3>尚硅谷北京校区</h3>
      <p>北京校区位于宏福科技园区</p>
    </div>
  </div>
</body>
</html>
"""

# 4.用于指定要根据哪些HTML标签来分割文本
headers_to_split_on = [
  ("h1", "标题1"),
  ("h2", "标题2"),
  ("h3", "标题3"),
]
# 5.定义HTMLHeaderTextSplitter分割器
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# 6.分割器分割
html_header_splits = html_splitter.split_text(html_string)
html_header_splits
```

说明：
- 标题下文本内容所属标题的层级信息保存在元数据中。
- 每个分块会自动继承父级标题的上下文，避免信息割裂。

**类型2：CodeTextSplitter：Split code**

CodeTextSplitter是一个 专为代码文件 设计的文本分割器（Text Splitter），支持代码的语言包括['cpp','go', 'java', 'js', 'php', 'proto', 'python', 'rst', 'ruby', 'rust', 'scala', 'swift', 'markdown', 'latex', 'html','sol']。它能够根据编程语言的语法结构（如函数、类、代码块等）智能地拆分代码，保持代码逻辑的完整性。

与递归文本分割器（如RecursiveCharacterTextSplitter）不同，CodeTextSplitter 针对代码的特性进行了优化，**`避免在函数或类的中间截断`**。

**举例**：支持的语言

> pip install langchain-text-splitters

```python
from langchain.text_splitter import Language
# 支持分割语言类型
# Full list of supported languages
langs = [e.value for e in Language]
print(langs)
```

**类型3：MarkdownTextSplitter：md数据类型**

因为Markdown格式有特定的语法，一般整体内容由 **h1、h2、h3** 等多级标题组织，所以
MarkdownHeaderTextSplitter的切分策略就是根据 **`标题来分割文本内容`**。

**举例：**

```python
from langchain.text_splitter import MarkdownTextSplitter
markdown_text = """
# 一级标题\n
这是一级标题下的内容\n\n
## 二级标题\n
- 二级下列表项1\n
- 二级下列表项2\n
"""

# 关键步骤：直接修改实例属性
splitter = MarkdownTextSplitter(chunk_size=30, chunk_overlap=0)
splitter.is_separator_regex = True # 强制将分隔符视为正则表达式
# 执行分割
docs = splitter.create_documents(texts = [markdown_text])
# print(len(docs))
for i, doc in enumerate(docs):
  print(f"\n🔍 分块{i + 1}:")
  print(doc.page_content)
```

## 4. 文档嵌入模型 Text Embedding Models
### 4.1 嵌入模型概述
**Text Embedding Models**：文档嵌入模型，提供将文本编码为向量的能力，即 **`文档向量化`** 。 `文档写入` 和 `用户查询匹配` 前都会先执行文档嵌入编码，即向量化。

![alt text](/public/langchain/retrieval/4.png)

- LangChain提供了 超过25种 不同的嵌入提供商和方法的集成，从开源到专有API，总有一款适合你。
- Hugging Face等开源社区提供了一些文本向量化模型（例如BGE），效果比闭源且调用API的向量化模型效果好，并且向量化模型参数量小，在CPU上即可运行。所以，这里推荐在开发RAG应用的过程中，使用 开源的文本向量化模型 。此外，开源模型还可以根据应用场景下收集的数据对模型进行微调，提高模型效果。

LangChain中针对向量化模型的封装提供了两种接口，一种针对 **`文档的向量化(embed_documents)`** ，一种针对 **`句子的向量化embed_query`**。

### 4.2 句子的向量化（embed_query）

**举例：**

```python
from langchain
import os
import dotenv_openai import OpenAIEmbeddings
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
# 初始化嵌入模型
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
#embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
# 待嵌入的文本句子
text = "What was the name mentioned in the conversation?"
# 生成一个嵌入向量
embedded_query = embeddings_model.embed_query(text = text)
# 使用embedded_query[:5]来查看前5个元素的值
print(embedded_query[:5])
print(len(embedded_query))
```
### 4.3 文档的向量化（embed_documents）
文档的向量化，接收的参数是字符串数组。

**举例1：**
```python
from langchain_openai import OpenAIEmbeddings
import numpy as np
import pandas as pd
import os
import dotenv
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
# 初始化嵌入模型
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
# 待嵌入的文本列表
texts = [
  "Hi there!",
  "Oh, hello!",
  "What's your name?",
  "My friends call me World",
  "Hello World!"
]
# 生成嵌入向量
embeddings = embeddings_model.embed_documents(texts)
for i in range(len(texts)):
  print(f"{texts[i]}:{embeddings[i][:3]}",end="\n\n")
```

**举例2：**
```python
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings(
  model="text-embedding-3-large",
)
# 情况1：
loader = CSVLoader("./asset/load/03-load.csv", encoding="utf-8")
docs = loader.load_and_split()
#print(len(docs))
# 存放的是每一个chrunk的embedding
embeded_docs = embeddings_model.embed_documents([doc.page_content for doc in docs])
print(len(embeded_docs))
# 表示的是每一个chrunk的embedding的维度
print(len(embeded_docs[0]))
print(embeded_docs[0][:10])
```

## 5. 向量存储(Vector Stores)
### 5.1 理解向量存储
将文本向量化之后，下一步就是进行向量的存储。这部分包含两块：
- `向量的存储` ：将非结构化数据向量化后，完成存储
- `向量的查询` ：查询时，嵌入非结构化查询并检索与嵌入查询“最相似”的嵌入向量。即具有相似性检索能力

![alt text](/public/langchain/retrieval/5.png)

![alt text](/public/langchain/retrieval/6.png)

### 5.2 常用的向量数据库

Langchain提供了超过50种不同向量存储(Vetor Stores)的集成，从开源的本地向量存储到云托管的私有向量存储，允许你选择最适合需求的向量存储。

LangChain支持的向量存储参考 `VetorStore` 接口和实现。

典型的介绍如下：
|向量数据库|描述|
|:---:|:---:|
|Chroma|开源、免费的嵌入式数据库|
|FAISS|Meta出品、开源、免费、Facebook AI相似性搜索服务。(Facebook AI Similarity Search,Facebook AI相似性搜索库)|
|Milvus|用于存储、索引和管理由深度神经网络和其他ML模型产生的大量嵌入向量的数据库|
|Pinecone|用于广泛功能的向量数据库|
|Redis|基于Redis的检索器|

### 5.3 向量数据库的理解
假设你是一名摄影师，拍了大量的照片。为了方便管理和查找，你决定将这些 照片存储 到一个数据库中。传统的 关系型数据库 （如 MySQL、PostgreSQL 等）可以帮助你 存储照片的元数据 ，比如拍摄时间、地点、相机型号等。

但是，当你想要根据 `照片的内容（如颜色、纹理、物体等）` 进行搜索时，传统数据库将无法满足你的需求，因为它们通常以数据表的形式存储数据，并使用查询语句进行精确搜索。那么此时，向量数据库就可以派上用场。

我们可以构建一个多维的空间使得每张照片特征都存在于这个空间内，并用已有的维度进行表示，比如时间、地点、相机型号、颜色....此照片的信息将作为一个点，存储于其中。以此类推，即可在该空间中构建出无数的点，而后我们将这些点与空间坐标轴的原点相连接，就成为了一条条向量，当这些点变为向量之后，即可利用向量的计算进一步获取更多的信息。当要进行照片的检索时，也会变得更容易更快捷。

**注意**：在向量数据库中进行检索时，检索并 `不是唯一的、精确的` ，而是查询和目标向量 最为相似的一些向量 ，具有模糊性。

**延伸思考：** 只要对图片、视频、商品等素材进行向量化，就可以实现以图搜图、视频相关推荐、相似宝贝推荐等功能。

### 5.4 代码实现
使用向量数据库组件时需要同时传入包含 文本块的Document类对象 以及 `文本向量化组件` ，向量数据库组件会自动完成将文本向量化的工作，并写入数据库中。

#### 5.4.1 数据的存储

**举例**：从TXT文档中加载数据，向量化后存储到Chroma数据库

安装模块：
> pip install chromadb
> 
> pip install langchain-chroma

```python
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma_community.document
from langchain_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
# 举例：将分割后的文本，使用 OpenAI 嵌入模型获取嵌入向量，并存储在 Chroma 中
# 获取嵌入模型
my_embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
# 创建TextLoader实例，并加载指定的文档
loader = TextLoader("./asset/load/09-ai1.txt", encoding='utf-8')
documents = loader.load()
# 创建文本拆分器
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# 拆分文档
docs = text_splitter.split_documents(documents)

# 存储：将文档和数据存储到向量数据库中
db = Chroma.from_documents(docs, my_embedding)
# 查询：使用相似度查找
query = "人工智能的核心技术都有啥？"
docs = db.similarity_search(query)
print(docs[0].page_content)
```

**思考：此时数据存储在哪里呢？**

**注意**：Chroma主要有两种存储模式： `内存模式` 和 `持久化模式` 。当使用persist_directory参数时，数据会保存到指定目录；如果没有指定，则默认使用内存存储。

#### 5.4.2 数据的检索
举例：一个包含构建Chroma向量数据库以及向量检索的代码
前置代码：
```python
# 1.导入相关依赖
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
# 2.定义文档
raw_documents = [
  Document(
    page_content="葡萄是一种常见的水果，属于葡萄科葡萄属植物。它的果实呈圆形或椭圆形，颜色有绿色、紫色、红色等多种。葡萄富含维生素C和抗氧化物质，可以直接食用或酿造成葡萄酒。",
    metadata={"source": "水果", "type": "植物"}
  ),
  Document(
    page_content="白菜是十字花科蔬菜，原产于中国北方。它的叶片形成紧密的球状层层包裹，口感清脆微甜。白菜富含膳食纤维和维生素K，常用于制作泡菜、炒菜或煮汤。",
    metadata={"source": "蔬菜", "type": "植物"}
  ),
  Document(
    page_content="狗是人类最早驯化的动物之一，属于犬科。它们具有高度社会性，能理解人类情绪，常被用作宠物、导盲犬或警犬。不同品种的狗在体型、毛色和性格上有很大差异。",
    metadata={"source": "动物", "type": "哺乳动物"}
  )

  # 3. 创建嵌入模型
embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
# 4.创建向量数据库
db = Chroma.from_documents(
  documents=raw_documents,
  embedding=embedding,
  persist_directory="./asset/chroma-3",
)
```

**① 相似性检索（similarity_search）**

接收字符串作为参数：

```python
# 5. 检索示例（返回前3个最相关结果）
query = "哺乳动物"
docs = db.similarity_
search(query, k=3) # k=3表示返回3个最相关文
print(f"查询: '{query}' 的结果:")
for i, doc in enumerate(docs, 1):
  print(f"\n结果 {i}:")
  print(f"内容: {doc.page_content}")
  print(f"元数据: {doc.metadata}")
```

**② 支持直接对问题向量查询（similarity_search_by_vector）**

搜索与给定嵌入向量相似的文档，它接受`嵌入向量作为参数`，而不是字符串。

```python
query = "哺乳动物"
embedding_vector = embedding.embed_query(query)

docs = db.similarity_search_by_vector(embedding_vector, k=3)

print(f"查询: '{query}' 的结果:")
for i, doc in enumerate(docs, 1):
  print(f"\n结果 {i}:")
  print(f"内容: {doc.page_content}")
  print(f"元数据: {doc.metadata}")
```

**③ 相似性检索，支持过滤元数据（filter）**

```python
query = "哺乳动物"
docs = db.similarity_search(query=query,k=3,filter={"source": "动物"})
for i, doc in enumerate(docs, 1):
  print(f"\n结果 {i}:")
  print(f"内容: {doc.page_content}")
  print(f"元数据: {doc.metadata}")
```

**④ 通过L2距离分数进行搜索（similarity_search_with_score）**

说明：分数值越小，检索到的文档越和问题相似。分值取值范围：[0，正无穷]

```python

docs = db.similarity_search_with_score(
  "量子力学是什么?"
)
for doc, score in docs:
  print(f" [L2距离得分={score:.3f}]{doc.page_content} [{doc.metadata}]")
```

**⑤ 通过余弦相似度分数进行搜索（_similarity_search_with_relevance_scores）**

说明：分数值越接近1（上限），检索到的文档越和问题相似。

```python
docs = db._similarity_search_with_relevance_scores(
  "量子力学是什么?"
)
for doc, score in docs:
  print(f"* [余弦相似度得分={score:.3f}]{doc.page_content} [{doc.metadata}]")
```

**⑥ MMR（最大边际相关性，max_marginal_relevance_search）**

MMR 是一种平衡 `相关性` 和 `多样性` 的检索策略，避免返回高度相似的冗余结果。

```python
docs = db.max_marginal_relevance_search(
  query="量子力学是什么",
  lambda_mult=0.8, # 侧重相似性
)
print("🔍 关于【量子力学是什么】的搜索结果：")
print("=" * 50)
for i, doc in enumerate(docs):
  print(f"\n📖 结果 {i+1}:")
  print(f"📌 内容: {doc.page_content}")
  print(f"🏷 标签: {', '.join(f'{k}={v}' for k, v in doc.metadata.items())}")
```

参数说明： `lambda_mult` 参数值介于 0 到 1 之间，用于确定结果之间的多样性程度，其中 0 对应最大多样性，1 对应最小多样性。默认值为 0.5。

## 6. 检索器(召回器) Retrievers
### 6.1 介绍

从“向量存储组件”的代码实现5.4.2中可以看到，向量数据库本身已经包含了实现召回功能的函数方法(`similarity_search`)。该函数通过计算原始查询向量与数据库中存储向量之间的相似度来实现召回。LangChain还提供了 `更加复杂的召回策略` ，这些策略被集成在Retrievers（检索器或召回器）组件中。

Retrievers（检索器）是一种用于从大量文档中检索与给定查询相关的文档或信息片段的工具。检索器`不需要存储文档` ，只需要 `返回（或检索）文档` 即可。

![alt text](/public/langchain/retrieval/7.png)

Retrievers 的执行步骤：
步骤1：将输入查询转换为向量表示。

步骤2：在向量存储中搜索与查询向量最相似的文档向量（通常使用余弦相似度或欧几里得距离等度量方法）。

步骤3：返回与查询最相关的文档或文本片段，以及它们的相似度得分。

### 6.2 代码实现
Retriever 一般和 VectorStore 配套实现，通过 `as_retriever()` 方法获取。

举例：

```python
# 1.导入相关依赖
import os
import dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

dotenv.load_dotenv()

# 2.定义文档加载器
loader = TextLoader(file_path='./asset/load/09-ai1.txt',encoding="utf-8")
# 3.加载文档
documents = loader.load()
# 4.定义文本切割器
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# 5.切割文档
docs = text_splitter.split_documents(documents)
# 6.定义嵌入模型
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
embeddings = OpenAIEmbeddings(
  model="text-embedding-3-large"
)

# 7.将文档存储到向量数据库中
db = FAISS.from_documents(docs, embeddings)

# 8.从向量数据库中得到检索器
retriever = db.as_retriever()

# 9.使用检索器检索
docs = retriever.invoke("深度学习是什么？")

print(len(docs))
# 10.得到结果
for doc in docs:
  print(f"⭐{doc}")
```

### 6.3 使用相关检索策略

前置代码：
```python
# 1.导入相关依赖
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
# 2.定义文档
document_1 = Document(
  page_content="经济复苏：美国经济正在从疫情中强劲复苏，失业率降至历史低点！
)

document_2 = Document(
  page_content="基础设施：政府将投资1万亿美元用于修复道路、桥梁和宽带网络。",
)

document_3 = Document(
page_content="气候变化：承诺到2030年将温室气体排放量减少50%。",

documents = [document_1,document_2,document_3]

# 3.创建向量存储
embeddings = OpenAIEmbeddings(
  model="text-embedding-3-large"
)

# 4.将文档向量化，添加到向量数据库索引中，得到向量数据库对象
db = FAISS.from_documents(documents, embeddings)
```

**① 默认检索器使用相似性搜索**
```python
# 获取检索器
retriever = db.as_retriever(search_kwargs={"k": 4}) #这里设置返回的文档数

docs = retriever.invoke("经济政策")

for i, doc in enumerate(docs):
  print(f"\n结果 {i+1}:\n{doc.page_content}\n")

```

**② 分数阈值查询**

只有相似度超过这个值才会召回

```python
retriever = db.as_retriever(
  search_type="similarity_score_threshold",
  search_kwargs={"score_threshold": 0.1}
)

docs = retriever.invoke("经济政策")

for doc in docs:
  print(f"📌 内容: {doc.page_content}")

```
> 📌 内容: 经济复苏：美国经济正在从疫情中强劲复苏，失业率降至历史低点。！

注意只会返回满足阈值分数的文档，不会获取文档的得分。如果想查询文档的得分是否满足阈值，可以使用向量数据库的 similarity_search_with_relevance_scores 查看（在5.4.2 情况5中讲过）。

```python
docs_with_scores = db.similarity_search_with_relevance_scores("经济政策")
for doc, score in docs_with_scores:
  print(f"\n相似度分数: {score:.4f}")
  print(f"📌 内容: {doc.page_content}")

```

**③ MMR搜索**

```python
retriever = db.as_retriever(
  search_type="mmr",
  # search_kwargs={"fetch_k":2}
)

docs = retriever.invoke("经济政策")
print(len(docs))
for doc in docs:
  print(f"📌 内容: {doc.page_content}")
```

### 6.4 结合LLM

举例1：通过FAISS构建一个可搜索的向量索引数据库，并结合RAG技术让LLM去回答问题。

**情况1：不用RAG给LLM灌输上下文数据**

```python
from langchain_openai import ChatOpenAI
import os
import dotenv
dotenv.load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")

# 创建大模型实例
llm = ChatOpenAI(model="gpt-4o-mini")
# 调用

response = llm.invoke("北京有什么著名的建筑？")
print(response.content)

```

**情况2：使用RAG给LLM灌输上下文数据**

> pip install faiss-cpu

```python
# 1. 导入所有需要的包
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
import dotenv

dotenv.load_dotenv()
# 2. 创建自定义提示词模板
prompt_template = """请使用以下提供的文本内容来回答问题。仅使用提供的文本信息，如果文本中没有相关信息，请回答"抱歉，提供的文本中没有这个信息"。
文本内容：
{context}

问题：{question}

回答：
"
"""
prompt = PromptTemplate.from_template(prompt_template)
# 3. 初始化模型
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
llm = ChatOpenAI(
  model="gpt-4o-mini",
  temperature=0
)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# 4. 加载文档
loader = TextLoader("./asset/load/10-test_doc.txt", encoding='utf-8')
documents = loader.load()
# 5. 分割文档
text_splitter = CharacterTextSplitter(
  chunk_size=1000,
  chunk_overlap=100,
)
texts = text_splitter.split_documents(documents)
#print(f"文档个数:{len(texts)}")
# 6. 创建向量存储
vectorstore = FAISS.from_documents(
  documents=texts,
  embedding=embedding_model
)
# 7.获取检索器
retriever = vectorstore.as_retriever()

docs = retriever.invoke("北京有什么著名的建筑？")
# 8. 创建Runnable链
chain = prompt | llm
# 9. 提问
result = chain.invoke(input={"question":"北京有什么著名的建筑？","context":docs})
print("\n回答:", result.content)
```