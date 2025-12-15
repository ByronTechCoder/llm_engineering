import os
import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings



def fetch_hazbin_data():
    hazbin_base = Path(__file__).parent.parent / "knowledge-base"
    ljson_path = hazbin_base / "hazbin_character_profiles.ljson"
    documents = []
    if ljson_path.exists():
        with open(ljson_path, "r", encoding="utf-8") as f:
            # Each line is a JSON object
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # You can decide how to populate Document's .page_content and .metadata
                content = data.get("description") or data.get("bio") or json.dumps(data)
                # Use all keys except content as metadata
                metadata = {k: v for k, v in data.items() if k != "description" and k != "bio"}
                documents.append(Document(page_content=content, metadata=metadata))
    return documents


def create_chunks(documents):
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
   chunks = text_splitter.split_documents(documents)
   return chunks


def create_embeddings(chunks):
    # Use Chroma to store vector embeddings of chunks
    # You may want to use a consistent persist_directory for reuse
    persist_dir = str(Path(__file__).parent.parent / "hazbin_vector_db")
    # Remove any existing vector store if you want a fresh ingestion. You can comment this out if not needed.
    if os.path.exists(persist_dir):
        Chroma(persist_directory=persist_dir).delete_collection()

    # Create/Open the Chroma vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    # Optionally print some stats about the vector store
    collection = vectorstore._collection
    count = collection.count()
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"Stored {count} vectors in Chroma, each with {dimensions} dimensions.")
    return vectorstore
