# This program uses RAG, with ChromaDB. 
# It's data is being supported by a .txt file, 2008_Jeep_Compass_Limted_FWD.txt,
# that supports information about the 2008 Jeep Compass.

import os
from dotenv import load_dotenv
import custom_console # Assuming this is available and works as intended

# LlamaIndex
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core import SimpleDirectoryReader # <-- Crucial: For loading documents
from llama_index.core.node_parser import SentenceSplitter # Optional: For better chunking

# Chroma
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# Google Gemini Flash 1.5
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# Async support for top-level await
import asyncio

# 1. Load environment variables from the .env file
load_dotenv()

# 2. Get the API key value from the environment
google_api_key = os.getenv("GOOGLE_API_KEY")

# Set the API key for Google's libraries (Good practice)
os.environ["GOOGLE_API_KEY"] = google_api_key

# Initialize the Gemini 1.5 Flash LLM
llm = GoogleGenAI(model="gemini-1.5-flash")
embed_model = GoogleGenAIEmbedding()

# Set it as the default LLM and embedding model for LlamaIndex
Settings.llm = llm
Settings.embed_model = embed_model
# Settings.chunk_size = 1024 # Optional: control chunking if not using a specific SentenceSplitter in pipeline
# Settings.chunk_overlap = 20 # Optional

custom_console.clear_console()
custom_console.simple_spinner(duration=3)

async def main(): # Wrap the main logic in an async function

    # --- Step 1: Load Documents ---
    # Create a 'data' directory in the same location as your script
    # and place your text files (e.g., .txt, .pdf, .md) inside it.
    # IMPORTANT: Ensure your files contain information about American Revolution battles in NYC.
    try:
        documents = SimpleDirectoryReader(input_dir="./data").load_data()
        print(f"Loaded {len(documents)} documents from ./data")
        if not documents:
            print("WARNING: No documents found in './data'. Please add relevant text files.")
            return # Exit if no documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        print("Please ensure you have a 'data' directory with documents inside.")
        return

    # --- Step 2: Initialize ChromaDB and Ingest Documents ---
    db = chromadb.PersistentClient(path="./alfred_chroma_db")
    chroma_collection = db.get_or_create_collection("alfred")

    # Clear existing data in the collection if you want a fresh start each time
    # This is often useful during development to avoid stale data
    # print(f"Current items in collection before ingestion: {chroma_collection.count()}")
    # if chroma_collection.count() > 0:
    #     print("Deleting existing items from collection for a fresh start...")
    #     chroma_collection.delete(ids=[item.id for item in chroma_collection.get()["ids"]])
    #     print(f"Items after deletion: {chroma_collection.count()}")


    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # This is the crucial line for indexing documents into the vector store
    # If the collection is already populated from a previous run and you don't want to re-index,
    # you can skip `from_documents` and directly use `from_vector_store` if `chroma_collection.count() > 0`.
    # However, for a robust solution that handles both new and existing data,
    # `from_documents` with `vector_store` specified is generally preferred for initial setup/updates.
    
    # Check if the collection is empty. If it is, then build the index from documents.
    # Otherwise, load the index from the already populated vector store.
    if chroma_collection.count() == 0:
        print("ChromaDB collection 'alfred' is empty. Indexing documents...")
        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store,
            # transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=20)], # Optional: if you want custom chunking
            embed_model=embed_model # Explicitly pass the embed model
        )
        print(f"Successfully indexed {len(documents)} documents.")
    else:
        print(f"ChromaDB collection 'alfred' already contains {chroma_collection.count()} items. Loading existing index.")
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model # Must pass embed_model even when loading from existing store
        )


    # --- Step 3: Setup Query Engine and Evaluator ---
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize",
        verbose=True # Set to True for more detailed logging from LlamaIndex
    )

    evaluator = FaithfulnessEvaluator(llm=llm)

    # --- Step 4: Perform Query and Evaluation ---
    print("\n--- Running Query ---")
    try:
        response = await query_engine.aquery( # Use .aquery for async
            "What does the CVT transmission feel like to drive with?"
        )

        print("\n--- Query Response ---")
        # Access the string response using .response
        print(f"Response: {response.response}")
        
        print("\n--- Source Nodes (Retrieved Context) ---")
        if response.source_nodes:
            for i, node in enumerate(response.source_nodes):
                print(f"  Node {i+1} Score: {node.score:.4f}")
                print(f"  Node {i+1} Text (first 300 chars): {node.text[:300]}...")
        else:
            print("No source nodes retrieved. This indicates an issue with retrieval.")

        print("\n--- Running Faithfulness Evaluation ---")
        # FaithfulnessEvaluator expects a Response object, which `query_engine.aquery` returns
        eval_result = await evaluator.aevaluate_response(response=response) # Use .aevaluate_response for async

        print("\n--- Evaluation Result ---")
        print(f"Query: {eval_result.query}")
        print(f"Response: {eval_result.response}")
        print(f"Passing: {eval_result.passing}")
        print(f"Feedback: {eval_result.feedback}")
        print(f"Score: {eval_result.score}")

    except Exception as e:
        print(f"An error occurred during query or evaluation: {e}")
        print("Common causes: LLM API issues, network problems, or very poor data quality.")

if __name__ == "__main__":
    asyncio.run(main())