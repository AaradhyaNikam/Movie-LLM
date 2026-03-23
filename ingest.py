import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings

# Paste your Cohere Trial Key here
os.environ["COHERE_API_KEY"] = "COHERE_API_KEY"

PDF_FILES = ["Movies_A-F.pdf", "Movies_G-L.pdf", "Movies_M-R.pdf", "Movies_S-Z.pdf"]
VECTORSTORE_PATH = "faiss_index_movies"

print("🚀 Starting local embedding process with Cohere...")
embeddings = CohereEmbeddings(model="embed-english-v3.0")

# 1. Load PDFs
all_documents = []
for pdf in PDF_FILES:
    if os.path.exists(pdf):
        print(f"📄 Loading {pdf}...")
        loader = PyPDFLoader(pdf)
        all_documents.extend(loader.load())

# 2. Split Text
print("✂️ Splitting text into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(all_documents)

print(f"Total chunks created: {len(splits)}")
print("🧠 Generating vectors (With Token Rate Limit Protection)...")

# 3. Robust Batching with Retry Logic
batch_size = 50 # Smaller batch size to spread out token usage
vectorstore = None
total_batches = (len(splits) // batch_size) + 1

for i in range(0, len(splits), batch_size):
    batch = splits[i : i + batch_size]
    current_batch_num = (i // batch_size) + 1
    
    success = False
    while not success:
        try:
            print(f"   -> Processing batch {current_batch_num} of {total_batches}...")
            
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)
            
            success = True # If it worked, break the while loop
            time.sleep(8)  # Standard 8-second wait between batches
            
        except Exception as e:
            error_message = str(e).lower()
            # If the API complains about rate limits, pause and retry
            if "429" in error_message or "rate limit" in error_message:
                print("   ⏳ 100k Token Limit hit! Pausing for 60 seconds to let the bucket refill...")
                time.sleep(60) 
            else:
                # If it's a different error, stop the program
                raise e

# 4. Save the final database
vectorstore.save_local(VECTORSTORE_PATH)
print(f"✅ Success! Database safely generated and saved to {VECTORSTORE_PATH} folder.")