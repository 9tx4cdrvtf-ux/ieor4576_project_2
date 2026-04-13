"""Index rides info markdown files into ChromaDB for RAG retrieval."""

# What this script does:
#   1. Reads all .md files from the magic_items/ directory
#   2. Parses each file for the item name and metadata (type, rarity, attunement)
#   3. Embeds the full markdown text using ChromaDB's built-in embedding model
#   4. Stores the embeddings + metadata in a persistent ChromaDB collection
#
# This is a one-time offline step — run it once before starting the server.
# The output (chroma_data/) is a build artifact: gitignored locally, baked
# into the Docker image at build time.
#
# At query time, app.py opens the same chroma_data/ directory (read-only)
# and calls collection.query() to find items similar to the user's message.

"""Index Disney ride markdown files into ChromaDB for RAG retrieval."""

import os
import re
import chromadb

RIDES_DIR = "disney_rides_info"  
CHROMA_PATH = "chroma_data"
COLLECTION_NAME = "disney_rides_info" 


H1_PATTERN = re.compile(r"^#\s+(.+)")

import os
import re

H1_PATTERN = re.compile(r"^#\s+(.+)")
LOCATION_PATTERN = re.compile(r"\*\*Location:\*\*\s*(.+)")
HEIGHT_PATTERN = re.compile(r"minimum height requirement is\s*([\d\.]+)\s*(meters|inches)", re.IGNORECASE)

def parse_item(filepath: str) -> dict | None:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        return None

    filename_stem = os.path.splitext(os.path.basename(filepath))[0]
    lines = text.strip().split("\n")

    # 1. extract ride name (from H1 or fallback to filename)
    name = filename_stem
    for line in lines:
        m = H1_PATTERN.match(line.strip())
        if m:
            name = m.group(1).strip()
            break

    # 2. extract park name (from bolded "Location: " field)
    park_name = "Unknown Park"
    location_match = LOCATION_PATTERN.search(text)
    if location_match:
        park_name = location_match.group(1).strip()

    # 3. extract minimum height requirement (look for "minimum height requirement is X meters/inches")
    height_limit = 0.0
    height_match = HEIGHT_PATTERN.search(text)
    if height_match:
        height_limit = float(height_match.group(1))

    # 4. infer intensity level (simple keyword search for "roller coaster", "thrill-seekers", "inversions", "high-speed launch" = "High Thrill"; otherwise "Family")
    intensity = "Family"
    thrill_keywords = ["roller coaster", "thrill-seekers", "inversions", "high-speed launch"]
    if any(kw in text.lower() for kw in thrill_keywords):
        intensity = "High Thrill"

    return {
        "id": filename_stem,
        "document": text,
        "metadata": {
            "name": name,
            "park": park_name,
            "intensity": intensity,
            "min_height": height_limit,
            "source_file": filename_stem
        },
    }


def main():
    if not os.path.exists(RIDES_DIR):
        print(f"error: cannot find '{RIDES_DIR}'")
        return

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # get_or_create_collection uses ChromaDB's default embedding function:
    # all-MiniLM-L6-v2 (via sentence-transformers), which runs locally with
    # no API key or network call required. It's fast and good enough for demos.
    #
    # In production you'd likely use your LLM provider's embedding model instead
    # (e.g. Vertex AI text-embedding-005) for better quality and consistency.
    #
    # IMPORTANT: whatever embedding model you use at index time MUST be the same
    # model used at query time — otherwise the similarity scores are meaningless.
    collection = client.get_or_create_collection(COLLECTION_NAME)

    md_files = sorted(
        os.path.join(RIDES_DIR, f)
        for f in os.listdir(RIDES_DIR)
        if f.endswith(".md")
    )

    ids = []
    documents = []
    metadatas = []
    skipped = 0

    print(f"Starting to process {len(md_files)} files...")

    for filepath in md_files:
        item = parse_item(filepath)
        if item is None:
            skipped += 1
            continue
        ids.append(item["id"])
        documents.append(item["document"])
        metadatas.append(item["metadata"])

    BATCH_SIZE = 100
    for i in range(0, len(ids), BATCH_SIZE):
        collection.upsert(
            ids=ids[i : i + BATCH_SIZE],
            documents=documents[i : i + BATCH_SIZE],
            metadatas=metadatas[i : i + BATCH_SIZE],
        )

    print(f"Successfully indexed {len(ids)} items (skipped {skipped} empty files)")
    print(f"Database saved to: {CHROMA_PATH}")

if __name__ == "__main__":
    main()