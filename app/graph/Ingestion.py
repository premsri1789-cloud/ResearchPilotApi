import os
import re
import base64
from dotenv import load_dotenv
import pymupdf4llm
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType

from app.core.vector_db import qdrant_client


#import env api keys
load_dotenv()

#vision llm
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "llama-3.3-70b-versatile")
vision_llm = ChatGroq(model=VISION_MODEL, temperature=0)


#embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    output_dimensionality=1536
)

def setup_collection(COLLECTIONNAME: str) -> None:
    # IF you have access to Qdrant Cloud dashboard, just delete the collection there manually.
    # OR, update your setup_collection to handle re-creation:
    if qdrant_client.collection_exists(COLLECTIONNAME):
        info = qdrant_client.get_collection(COLLECTIONNAME)
        if info.config.params.vectors.size != 1536:
            print("Dimension mismatch! Deleting old collection...")
            qdrant_client.delete_collection(COLLECTIONNAME)
            
    # Now create it with the correct 1536 dimension AND create indexes
    if not qdrant_client.collection_exists(COLLECTIONNAME):
        qdrant_client.create_collection(
            collection_name=COLLECTIONNAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE) # 1536 for Gemini
        )
        print(f'Collection {COLLECTIONNAME} created with 1536 dimensions')
        
        # CRITICAL: Create indexes for filtering immediately after collection creation
        qdrant_client.create_payload_index(
            collection_name=COLLECTIONNAME,
            field_name="user_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        qdrant_client.create_payload_index(
            collection_name=COLLECTIONNAME,
            field_name="session_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print("Indexes created for user_id and session_id")

def encode_image(image_path: str) -> str:
    """Encodes an image to base64 string for the Vision Model."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def summarize_visual(image_path: str, element_type: str = "image or table") -> str:
    """Attempts to summarize an extracted visual with Groq and falls back gracefully if the model cannot handle image input."""
    prompt = (
        f"You are an expert data analyst. Describe this {element_type} in detail. "
        f"If it's a table, extract the key data points and trends. "
        f"If it's a flowchart or image, explain the relationships and meaning. "
        f"The visual file is saved at: {image_path}."
    )

    try:
        # Try a multimodal payload first when the chosen model appears vision-capable.
        if "vision" in VISION_MODEL.lower() or "llama-4" in VISION_MODEL.lower():
            base64_image = encode_image(image_path)
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ]
            )
            response = vision_llm.invoke([message])
            return response.content if hasattr(response, "content") else str(response)

        # Fallback for text-only models such as llama-3.3-70b-versatile.
        response = vision_llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        print(f"Groq visual summarization failed for {image_path}: {exc}. Falling back to a lightweight description.")
        return (
            f"Visual content was detected at {os.path.basename(image_path)}. "
            "Automatic AI summarization was skipped because the configured Groq model could not process the image input."
        )

def ingest_pipeline(file_path: str, session_id: str, user_id: str, COLLECTIONNAME: str): # Added user_id
    """Parses a PDF to Markdown, extracts images, adds AI meaning, and chunks by topic."""

    print(f"Converting {file_path} to Markdown and extracting visuals...")
    media_dir = "/tmp/extracted_media" # CHANGED TO /tmp FOR SERVERLESS
    os.makedirs(media_dir, exist_ok=True)
    
    # 1. PyMuPDF4LLM converts PDF directly to Markdown and saves images to the folder natively
    md_text = pymupdf4llm.to_markdown(doc=file_path, write_images=True, image_path=media_dir)
    
    # 2. Chunk based on Markdown Headers (Topical/Section Chunking)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = markdown_splitter.split_text(md_text)
    
    print(f'Created {len(chunks)} topical chunks. Scanning for visuals to summarize...')

    points = []
    for idx, chunk in enumerate(chunks):
        content_to_embed = chunk.page_content
        
        # 3. Use Regex to find any images saved within this specific topical chunk
        # Markdown images look like: ![alt text](path/to/image.png) or ![alt text](extracted_media/img.png)
        image_paths = re.findall(r"!\[.*?\]\((.*?)\)", content_to_embed)
        
        for img_path in image_paths:
            # Verify the image exists on disk
            if os.path.exists(img_path):
                print(f"--> Processing visual at {img_path} with Groq...")
                visual_summary = summarize_visual(img_path)
                
                # Append the AI's rich description to the chunk's content for vector search
                content_to_embed += f"\n\n[AI Visual Summary for {img_path}]: {visual_summary}"

        # Generate embedding for the combined text + visual summary
        embedding = embedding_model.embed_query(content_to_embed)

        # Package the payload
        points.append(
            PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "session_id": session_id,
                    "text": chunk.page_content,
                    "section": chunk.metadata.get("Header 1", "General")
                }
            )
    )

    # Upsert into Qdrant
    if points:
        qdrant_client.upsert(
            collection_name=COLLECTIONNAME,
            points=points
        )
        print(f"Successfully ingested {len(points)} vectors (with multimodal summaries) for session: {session_id}")
    else:
        print("No chunks were processed. Check if the PDF is empty.")


if __name__ == "__main__":
    COLLECTIONNAME = "Research_documents"
    setup_collection(COLLECTIONNAME) 

    if os.path.exists("research-paper.pdf"):
        ingest_pipeline("research-paper.pdf", "session_123", COLLECTIONNAME)
    else:
        print("The pdf file 'research-paper.pdf' does not exist. Please place a sample PDF in this directory.")