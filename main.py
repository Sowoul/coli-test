import os
import base64
from PIL import Image
from io import BytesIO
from byaldi import RAGMultiModalModel
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

# ========== Setup ==========
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load RAG model (CPU if GPU is too small)
rag_model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.3", device="cpu")

# ========== Utils ==========
def index_document(path: str, index_name="doc_index"):
    rag_model.index(
        input_path=path,
        index_name=index_name,
        store_collection_with_index=True,
        overwrite=True,
    )

def run_rag_search(query: str, k: int = 1):
    return rag_model.search(query, k=k, return_base64_results=True)

def call_llm(client, model: str, query: str, image_b64: str):
    return client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        }],
    )

# ========== Terminal-based Loop ==========
print("=== Multimodal RAG Terminal Assistant ===")
doc_path = input("Path to document (PDF / image): ").strip()
print("Indexing document...")
index_document(doc_path)
print("✅ Document indexed!")

while True:
    query = input("\nYour question (or 'exit' to quit): ").strip()
    if query.lower() == "exit":
        break

    results = run_rag_search(query)
    image_b64 = results[0].base64
    image = Image.open(BytesIO(base64.b64decode(image_b64)))

    with open('reference.png', 'wb') as f:
        f.write(base64.b64decode(image_b64))

    response = call_llm(client, "gpt-4o-mini", query, image_b64)
    output = response.choices[0].message.content

    print("\nAssistant:", output)
    image.show()  # opens the retrieved context image in your OS image viewer
