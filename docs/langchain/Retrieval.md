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
举例1：加载csv所有列

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

举例1：使用JSONLoader文档加载器加载

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

举例2：提取04-response.json文件中嵌套在 data.items[].content 的文本
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

**举例1：**

使用split_text()方法演示

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

**举例2：**

使用create_documents()方法演示，传入字符串列表，返回Document对象列表
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
