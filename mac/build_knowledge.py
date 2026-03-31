import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Define ABSOLUTE paths pointing directly to your mlx_env folder
BASE_DIR = os.path.expanduser("~/mlx_env")
KNOWLEDGE_DIR = BASE_DIR
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

print(f" Looking for datasets in: {KNOWLEDGE_DIR}")

print(" Initializing Embedding Model (Metal Accelerated)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print(" Loading Engineering Datasets...")
documents = []

# Load UIUC Airfoil Data (.dat files)
uiuc_path = os.path.join(KNOWLEDGE_DIR, "UIUC-Airfoil-Database")
if os.path.exists(uiuc_path):
    print(f" -> Loading UIUC Airfoils from {uiuc_path}...")
    # Some older airfoil files have weird text encodings, so we skip the ones that can't be read cleanly
    loader = DirectoryLoader(uiuc_path, glob="**/*.dat", loader_cls=TextLoader, loader_kwargs={'autodetect_encoding': True})
    try:
        documents.extend(loader.load())
    except Exception as e:
        print(f"    Minor warning while loading some UIUC files: {e}")
else:
    print(f"  Could not find UIUC path: {uiuc_path}")

# Load AeroSandbox Examples (.py files)
asb_path = os.path.join(KNOWLEDGE_DIR, "AeroSandbox", "tutorial") 
if os.path.exists(asb_path):
    print(f" -> Loading AeroSandbox Tutorials from {asb_path}...")
    loader = DirectoryLoader(asb_path, glob="**/*.py", loader_cls=TextLoader)
    documents.extend(loader.load())
else:
    print(f"  Could not find AeroSandbox path: {asb_path}")

print(f"\n Loaded {len(documents)} raw engineering files.")

# Safety Check
if len(documents) == 0:
    print(" Error: No documents found! Exiting before ChromaDB crashes.")
    exit()

# 2. Chunk the documents
print(" Chunking documents into digestible pieces...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_documents(documents)
print(f" Created {len(chunks)} searchable chunks.")

# 3. Build and Save the Vector Database
print(" Embedding chunks and building ChromaDB (This may take a minute or two)...")
vectorstore = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory=CHROMA_DB_DIR
)

print(f"\n Success! Your RAG knowledge base is built and saved to {CHROMA_DB_DIR}")
