import os
import re
import base64
from dotenv import load_dotenv
import pymupdf4llm
from fastembed import TextEmbedding
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.core.vector_db import qdrant_client

load_dotenv()

VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "llama-3.3-70b-versatile")
vision_llm = ChatGroq(model=VISION_MODEL, temperature=0)

# Local FastEmbed (384 dimensions)
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def setup_collection(COLLECTIONNAME: str) -> None:
    if qdrant_client.collection_exists(COLLECTIONNAME):
        info = qdrant_client.get_collection(COLLECTIONNAME)
        if info.config.params.vectors.size != 384:
            qdrant_client.delete_collection(COLLECTIONNAME)
            
    if not qdrant_client.collection_exists(COLLECTIONNAME):
        qdrant_client.create_collection(
            collection_name=COLLECTIONNAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"Collection '{COLLECTIONNAME}' initialized locally (384 dim).")

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def summarize_visual(image_path: str, element_type: str = "image or table") -> str:
    prompt = f"Describe this {element_type} in detail. Extract key points or trends."
    try:
        if "vision" in VISION_MODEL.lower():
            base64_image = encode_image(image_path)
            message = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ])
            return vision_llm.invoke([message]).content
        return vision_llm.invoke(prompt).content
    except Exception:
        return f"Visual content processed at {os.path.basename(image_path)}."

def ingest_pipeline(file_path: str, session_id: str, user_id: str, COLLECTIONNAME: str):
    print(f"Ingesting file: {file_path}")
    media_dir = "./extracted_media"
    os.makedirs(media_dir, exist_ok=True)
    
    md_text = pymupdf4llm.to_markdown(doc=file_path, write_images=True, image_path=media_dir)
    
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = markdown_splitter.split_text(md_text)
    
    points = []
    for idx, chunk in enumerate(chunks):
        content_to_embed = chunk.page_content
        image_paths = re.findall(r"!\[.*?\]\((.*?)\)", content_to_embed)
        
        for img_path in image_paths:
            if os.path.exists(img_path):
                visual_summary = summarize_visual(img_path)
                content_to_embed += f"\n\n[Visual Summary]: {visual_summary}"

        embeddings_generator = embedding_model.embed([content_to_embed])
        embedding = list(embeddings_generator)[0].tolist()

        points.append(
            PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "session_id": session_id,
                    "user_id": user_id, 
                    "text": chunk.page_content
                }
            )
        )

    if points:
        qdrant_client.upsert(collection_name=COLLECTIONNAME, points=points)
        print(f"Ingested {len(points)} vectors locally.")