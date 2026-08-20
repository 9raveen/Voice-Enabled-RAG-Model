"""
Phase 5, Step 3 — Full pipeline test: real query -> live Qdrant
retrieval -> generation harness -> validated answer. No hardcoded data.
"""

from retrieval import retrieve
from generation import generate_answer

TEST_QUERIES = [
    "कॉर्पोरेशन क्या है?",
    "महात्मा गांधी का जन्म कब हुआ था?",
]

for query in TEST_QUERIES:
    print(f"\n{'='*60}")
    print(f"Query: {query}")

    retrieval_result = retrieve(query)
    print(f"Top score: {retrieval_result.top_score:.4f} | Confident: {retrieval_result.confident}")
    print(f"Retrieved {len(retrieval_result.chunks)} chunks")
    if retrieval_result.chunks:
        print(f"Top chunk: {retrieval_result.chunks[0].text[:100]}...")

    answer = generate_answer(retrieval_result)
    print(f"\nStatus: {answer.status}")
    print(f"Answer: {answer.answer}")