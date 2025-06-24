from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.tools.google import GmailToolSpec

import chromadb
import custom_console

custom_console.clear_console()
custom_console.simple_spinner(duration=3)

tool_spec = GmailToolSpec()
tool_spec_list = tool_spec.to_tool_list()

embed_model = GoogleGenAIEmbedding("gemini-1.5-flash")

db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

llm = GoogleGenAI(model_name="gemini-1.5-flash")
query_engine = index.as_query_engine(llm=llm)
tool = QueryEngineTool.from_defaults(
    query_engine, name="some useful name", 
    description="some useful description"
)