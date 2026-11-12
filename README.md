# MediFood AI (Flask)

RAG-based health & nutrition chatbot for disease-specific Indian diet plans.
TF-IDF + cosine similarity retrieval. **No API key, no internet needed to run.**

## Run (3 steps)
```bash
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:5000
```
The vector store is already built and included in `vector_store/`.
To rebuild it after editing the CSVs: `python build_vector_store.py`
(or just delete `vector_store/` - the app rebuilds it automatically on start).

## Structure
```
medifood_flask/
├── app.py                  Flask app: routes, auth, /api/chat
├── build_vector_store.py   One-time "training" / indexing script
├── requirements.txt
├── data/                   13 CSV datasets (pre-loaded by developer)
├── vector_store/           TF-IDF vectorizer + sparse matrix + chunks
├── utils/
│   ├── data_loader.py      CSV rows -> text chunks
│   ├── embeddings.py       TF-IDF vectorisation (sparse)
│   ├── retriever.py        Query expansion + cosine similarity top-K
│   ├── answer_generator.py Disease-specific Indian diet plan
│   └── scope_checker.py    Health-only filter (off-topic -> out of scope)
├── templates/              login, signup, home, chat (Jinja2)
├── static/style.css
└── instance/users.json     Created on first signup (hashed passwords)
```

## Flow
Signup/Login -> Home -> Chat. Each question: scope check -> query expansion ->
TF-IDF vector -> cosine similarity top-5 -> disease-aware structured answer.

Educational mini-project. Always consult a doctor for medical advice.
Set `SECRET_KEY` env var before any real deployment.
