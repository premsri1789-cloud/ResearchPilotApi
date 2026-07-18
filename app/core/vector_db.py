import os
from qdrant_client import QdrantClient

# Explicitly use local storage for your local environment
qdrant_client = QdrantClient(path="./local_quadrant_db")