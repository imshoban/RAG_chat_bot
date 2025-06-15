import docx
import PyPDF2
import os
from semantic_search import semantic_search
from chromadb_config import collection
import boto3
import json

# Create Bedrock Runtime client
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')



def read_docx_file(file_path: str):
    doc = docx.Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])


def split_text(text: str, chunk_size: int = 500):
    """Split text into chunks while preserving sentence boundaries"""
    try:
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Ensure proper sentence ending
            if not sentence.endswith('.'):
                sentence += '.'

            sentence_size = len(sentence)

            # Check if adding this sentence would exceed chunk size
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks
    except Exception as e:
        print('exception as ', e)


def process_document(file_path: str):
    """Process a single document and prepare it for ChromaDB"""
    try:
        # Read the document
        content = read_docx_file(file_path)
        # Split into chunks
        chunks = split_text(content)

        # Prepare metadata
        file_name = os.path.basename(file_path)
        metadatas = [{"source": file_name, "chunk": i} for i in range(len(chunks))]
        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
        return ids, chunks, metadatas
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return [], [], []



def add_to_collection(collection, ids, texts, metadatas):
    """Add documents to collection in batches"""
    if not texts:
        return

    batch_size = 100
    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        collection.add(
            documents=texts[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )

def process_and_add_documents(collection, folder_path: str):
    """Process all documents in a folder and add to collection"""
    files = [os.path.join(folder_path, file) 
             for file in os.listdir(folder_path) 
             if os.path.isfile(os.path.join(folder_path, file))]

    for file_path in files:
        # print(f"Processing {os.path.basename(file_path)}...")
        ids, texts, metadatas = process_document(file_path)
        add_to_collection(collection, ids, texts, metadatas)
        # print(f"Added {len(texts)} chunks to collection")

def get_context_with_sources(results):
    """Extract context and source information from search results"""
    # Combine document chunks into a single context
    context = "\n\n".join(results['documents'][0])

    # Format sources with metadata
    sources = [
        f"{meta['source']} (chunk {meta['chunk']})" 
        for meta in results['metadatas'][0]
    ]

    return context, sources


def generate_response(query: str, context: str, conversation_history: str = ""):
    """Generate a response using OpenAI with conversation history"""
    payload = {
        "anthropic_version": "bedrock-2023-05-31",  
        "messages": [
            {
                "role": "user",
                "content": "You are a helpful assistant that answers questions based on the provided context. context from document" + context + query
            } 
        ],
        "max_tokens": 300,
        "temperature": 0.7
    }

    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",  
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json"
    )

    response_body = json.loads(response['body'].read())
    return response_body["content"][0]["text"]


folder_path = "./KB"
process_and_add_documents(collection, folder_path)
query = "What is the capital of india?"
results = semantic_search(collection, query)
context, sources = get_context_with_sources(results)
# print('context!!', context)
# print('source>>>', sources)
response = generate_response(query, context)
print('response - ', response)

