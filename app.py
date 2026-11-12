"""
app.py - MediFood AI (Flask)
Run:  python app.py   ->  http://127.0.0.1:5000
Routes: /login /signup /logout /home /chat /api/chat
"""
import os, sys, json, threading
from functools import wraps
from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, flash)
from werkzeug.security import generate_password_hash, check_password_hash
import markdown

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from utils.embeddings       import (build_tfidf_embeddings, save_embeddings,
                                    load_embeddings, is_vector_store_ready)
from utils.retriever        import retrieve_top_k
from utils.answer_generator import generate_answer
from utils.scope_checker    import is_health_query, OUT_OF_SCOPE_RESPONSE

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "medifood-dev-secret-change-me")

USERS_FILE = os.path.join(BASE, "instance", "users.json")
_lock = threading.Lock()

# ── Knowledge base (built automatically on first run if missing) ─────────────
if not is_vector_store_ready():
    print("Vector store missing - building it now (needs pandas) ...")
    from utils.data_loader import load_all_data
    _chunks = load_all_data(os.path.join(BASE, "data"))
    _vec, _mat = build_tfidf_embeddings(_chunks)
    save_embeddings(_vec, _mat, _chunks)
VECTORIZER, MATRIX, CHUNKS = load_embeddings()


# ── Simple JSON user store ───────────────────────────────────────────────────
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE) as f:
        return json.load(f)

def save_users(users):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def login_required(view):
    @wraps(view)
    def wrapped(*a, **kw):
        if "user" not in session:
            return redirect(url_for("login"))
        return view(*a, **kw)
    return wrapped


# ── Pages ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return redirect(url_for("home" if "user" in session else "login"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name  = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        pwd   = request.form.get("password", "")
        pwd2  = request.form.get("confirm", "")
        if not name or not email or not pwd:
            flash("Please fill in all fields.", "error")
        elif "@" not in email:
            flash("Please enter a valid email address.", "error")
        elif len(pwd) < 6:
            flash("Password must be at least 6 characters.", "error")
        elif pwd != pwd2:
            flash("Passwords do not match.", "error")
        else:
            with _lock:
                users = load_users()
                if email in users:
                    flash("An account with this email already exists.", "error")
                else:
                    users[email] = {"name": name, "email": email,
                                    "password": generate_password_hash(pwd)}
                    save_users(users)
                    flash("Account created! Please sign in.", "success")
                    return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        pwd   = request.form.get("password", "")
        user  = load_users().get(email)
        if not user:
            flash("No account found with this email.", "error")
        elif not check_password_hash(user["password"], pwd):
            flash("Incorrect password.", "error")
        else:
            session["user"] = {"name": user["name"], "email": email}
            return redirect(url_for("home"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/home")
@login_required
def home():
    return render_template("home.html", user=session["user"],
                           n_chunks=len(CHUNKS))

@app.route("/chat")
@login_required
def chat():
    return render_template("chat.html", user=session["user"])


# ── Chat API (the RAG pipeline) ──────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    data  = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    prof  = data.get("profile") or {}
    top_k = 5
    if not query:
        return jsonify(error="Please type a question."), 400

    # Step 1: scope check
    if not is_health_query(query):
        return jsonify(html=markdown.markdown(OUT_OF_SCOPE_RESPONSE),
                       sources=[], in_scope=False)

    # Step 2: retrieve   Step 3: generate
    retrieved = retrieve_top_k(query, VECTORIZER, MATRIX, CHUNKS, top_k=top_k)
    patient = {
        "Age": prof.get("age"), "Gender": prof.get("gender"),
        "Weight": f'{prof.get("weight")} kg' if prof.get("weight") else None,
        "Height": f'{prof.get("height")} cm' if prof.get("height") else None,
        "BMI": prof.get("bmi"),
        "Known conditions": ", ".join(prof.get("conditions") or []) or "Not specified",
        "Allergies": prof.get("allergies") or "None",
    }
    answer = generate_answer(query, retrieved, patient_profile=patient)
    return jsonify(html=markdown.markdown(answer),
                   sources=[{"source": r["source"], "score": round(r["score"], 3),
                             "text": r["text"]} for r in retrieved],
                   in_scope=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
