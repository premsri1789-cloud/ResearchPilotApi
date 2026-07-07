from qdrant_client import QdrantClient
import os

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# Initialize the Qdrant client ONCE here.
# This acts as a singleton. When other files import `qdrant_client`,
# Python uses this exact same instance instead of trying to acquire a second lock.
if QDRANT_URL and QDRANT_API_KEY:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    qdrant_client = QdrantClient(path="/tmp/local_quadrant_db")