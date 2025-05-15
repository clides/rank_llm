# How to Use External Integrations of RankLLM

## [LangChain](https://github.com/langchain-ai/langchain)
Set up the base vector store retriever:
```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
import os

torch.cuda.set_device(7)
os.environ["CUDA_VISIBLE_DEVICES"] = str(7)
device = "cuda"

documents = TextLoader("state_of_the_union.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
for idx, text in enumerate(texts):
    text.metadata["id"] = idx

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

retriever = FAISS.from_documents(texts, embedding).as_retriever(search_kwargs={"k": 20})
```

## [Rerankers](https://github.com/AnswerDotAI/rerankers)

## [Llama Index](https://github.com/run-llama/llama_index)