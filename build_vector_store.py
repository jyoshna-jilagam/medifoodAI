"""
build_vector_store.py
---------------------
This is the ONE-TIME "training" / indexing script.

Run this BEFORE launching the chatbot:
    python build_vector_store.py

What it does:
  1. Loads all CSV files from the data/ folder.
  2. Converts each row into a text chunk.
  3. Computes TF-IDF embeddings (converts text → numbers).
  4. Saves the vectorizer + embedding matrix to vector_store/.

After this runs, the chatbot can answer questions instantly
without re-reading the CSVs every time.

Think of this like "training" — but instead of deep learning,
we're building a fast lookup index using TF-IDF math.
"""

import os
import sys
import time

# Make sure Python can find our utils/ modules
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader  import load_all_data
from utils.embeddings   import (build_tfidf_embeddings,
                                save_embeddings,
                                is_vector_store_ready)


def main():
    print("=" * 55)
    print("  MediFood AI — Vector Store Builder")
    print("=" * 55)

    # ── Check if already built ────────────────────────────────────────────
    if is_vector_store_ready():
        print("\n✅ Vector store already exists. Skipping rebuild.")
        print("   Delete vector_store/ folder to rebuild from scratch.\n")
        return

    start = time.time()

    # ── Step 1: Load all CSV data ─────────────────────────────────────────
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    chunks   = load_all_data(data_dir)

    if not chunks:
        print("❌ No chunks loaded. Check that data/ folder has CSV files.")
        return

    # ── Step 2: Build TF-IDF embeddings ──────────────────────────────────
    print("⚙  Converting text chunks to TF-IDF vectors ...")
    vectorizer, matrix = build_tfidf_embeddings(chunks)

    # ── Step 3: Save everything to disk ──────────────────────────────────
    print("\n💾 Saving vector store ...")
    save_embeddings(vectorizer, matrix, chunks)

    elapsed = time.time() - start
    print(f"\n✅ Done! Vector store built in {elapsed:.1f} seconds.")
    print(f"   Total chunks indexed: {len(chunks)}")
    print(f"   Matrix size: {matrix.shape}")
    print("\n▶  Now run:  python app.py\n")


if __name__ == "__main__":
    main()
