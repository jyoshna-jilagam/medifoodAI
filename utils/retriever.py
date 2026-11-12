"""
retriever.py
------------
Step 3 of the RAG pipeline: Retrieve the most relevant chunks for a query.

How retrieval works (simple explanation):
  1. We have a big table (matrix) where each ROW = one text chunk,
     each COLUMN = one word in our vocabulary.
     The values are TF-IDF scores (importance of that word in that chunk).

  2. We convert the user's query into the SAME kind of row vector.

  3. We compute COSINE SIMILARITY between the query vector and
     every chunk vector.

  4. Cosine Similarity measures the ANGLE between two vectors.
     - Score = 1.0  → identical meaning
     - Score = 0.0  → completely different
     - Score = -1.0 → opposite (not possible with TF-IDF since values ≥ 0)

  5. We return the top-K chunks with the highest similarity scores.
     These are our "retrieved context".

Formula:
  cosine_similarity(A, B) = (A · B) / (|A| × |B|)

  Since we already L2-normalized our vectors in embeddings.py,
  |A| = |B| = 1, so:
  cosine_similarity = A · B  (just a dot product!)
"""

import numpy as np
from utils.embeddings import embed_query


# Disease keyword expansion map — adds extra terms so TF-IDF can match better
QUERY_EXPANSION = {
    "diabetes":     "diabetes diabetic blood sugar glucose insulin glycemic diet carbohydrate",
    "high bp":      "hypertension blood pressure sodium salt DASH diet potassium",
    "hypertension": "hypertension blood pressure sodium salt DASH diet potassium",
    "heart":        "heart disease cholesterol cardiovascular omega-3 saturated fat",
    "obesity":      "obesity weight loss calorie deficit fiber protein lean",
    "kidney":       "kidney renal low potassium low phosphorus low protein",
    "liver":        "liver hepatic fatty liver antioxidants low fat",
    "anemia":       "anemia iron deficiency vitamin C iron-rich foods spinach lentils",
    "thyroid":      "thyroid iodine selenium metabolism",
    "pcod":         "PCOS PCOD hormonal low glycemic anti-inflammatory",
}

GOOD_SOURCES = {
    "diabetes":     ["diet_recommendations_dataset", "Personalized_Diet_Recommendations",
                     "disease_diets", "healthy_eating_dataset", "Food_Nutrition_Dataset"],
    "hypertension": ["diet_recommendations_dataset", "Personalized_Diet_Recommendations",
                     "disease_diets", "healthy_eating_dataset"],
    "heart":        ["diet_recommendations_dataset", "Personalized_Diet_Recommendations",
                     "disease_diets", "healthy_meal_plans"],
    "general":      ["Food_Nutrition_Dataset", "healthy_eating_dataset",
                     "healthy_meal_plans", "Food_and_Calories"],
}


def expand_query(query_text):
    """Add disease-specific terms to improve TF-IDF retrieval."""
    q = query_text.lower()
    expanded = query_text
    for keyword, expansion in QUERY_EXPANSION.items():
        if keyword in q:
            expanded += " " + expansion
    return expanded


def retrieve_top_k(query_text, vectorizer, matrix, chunks, top_k=5):
    """
    Given a query string, find the top_k most relevant chunks.

    Parameters
    ----------
    query_text  : str   – the user's question
    vectorizer  : fitted TfidfVectorizer
    matrix      : numpy array (n_chunks × vocab_size), L2-normalized
    chunks      : list of dicts [{"text":..., "source":...}]
    top_k       : int   – how many chunks to return

    Returns
    -------
    results : list of dicts, each with keys:
              "text", "source", "score" (similarity 0-1)
    """
    # Step 1: Expand the query with disease-specific terms
    expanded_query = expand_query(query_text)

    # Step 2: Embed the expanded query
    query_vec = embed_query(expanded_query, vectorizer)  # shape (1, vocab_size)

    # Step 3: Compute dot product with all chunk vectors → similarity scores
    scores = matrix.dot(query_vec.T).toarray().ravel()

    # Step 4: Boost scores for high-quality nutrition/diet sources
    PRIORITY_SOURCES = [
        "diet_recommendations_dataset",
        "Personalized_Diet_Recommendations",
        "disease_diets",
        "healthy_eating_dataset",
        "healthy_meal_plans",
        "Food_Nutrition_Dataset",
        "Food_and_Calories",
    ]
    for i, chunk in enumerate(chunks):
        if chunk["source"] in PRIORITY_SOURCES:
            scores[i] *= 1.4   # 40% boost for nutrition-focused sources

    # Step 5: Get top_k results, fetch a bigger pool first then filter
    top_indices = np.argsort(scores)[::-1][:top_k * 3]

    # Step 6: Deduplicate by source to get diverse results
    seen_sources = {}
    results = []
    for idx in top_indices:
        if scores[idx] < 0.01:
            continue
        source = chunks[idx]["source"]
        # Allow max 2 per source to keep variety
        if seen_sources.get(source, 0) < 2:
            results.append({
                "text":   chunks[idx]["text"],
                "source": source,
                "score":  float(scores[idx]),
            })
            seen_sources[source] = seen_sources.get(source, 0) + 1
        if len(results) >= top_k:
            break

    return results


def format_context(retrieved_chunks):
    """
    Join the retrieved chunks into a single context string
    that will be passed to the answer generator.
    """
    if not retrieved_chunks:
        return "No relevant information found in the knowledge base."

    lines = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        lines.append(f"[{i}] (Source: {chunk['source']}, Score: {chunk['score']:.3f})")
        lines.append(chunk["text"])
        lines.append("")

    return "\n".join(lines)
