"""
scope_checker.py
----------------
Determines whether a user query is health/food related.
If it's off-topic, the chatbot should politely decline.

Uses simple keyword matching — no ML needed for scope checking.
"""

HEALTH_KEYWORDS = [
    # Diseases & conditions
    "diabetes", "diabetic", "hypertension", "blood pressure", "bp",
    "heart", "cardiac", "cholesterol", "obesity", "obese", "weight",
    "kidney", "renal", "liver", "thyroid", "anaemia", "anemia", "pcos",
    "pcod", "pregnancy", "pregnant", "cancer", "arthritis", "asthma",
    "covid", "flu", "fever", "infection", "vitamin", "mineral",
    "blood sugar", "glucose", "insulin", "creatinine", "haemoglobin",
    "hemoglobin", "tsh", "hba1c", "bmi", "calories", "nutrition",
    # Food & diet
    "food", "diet", "eat", "eating", "meal", "breakfast", "lunch",
    "dinner", "snack", "fruit", "vegetable", "protein", "carb",
    "fat", "fibre", "fiber", "nutrients", "recipe", "cook",
    "healthy", "unhealthy", "supplement", "vitamin", "iron",
    "calcium", "potassium", "sodium", "sugar", "carbohydrate",
    # Indian foods (common)
    "dal", "roti", "rice", "sabzi", "dosa", "idli", "upma", "poha",
    "khichdi", "chapati", "paratha", "rajma", "chhole", "palak",
    "methi", "karela", "lauki", "amla", "ragi", "bajra", "jowar",
    "ghee", "curd", "chaas", "paneer", "lassi", "milk", "dahi",
    # Health actions
    "exercise", "workout", "walk", "sleep", "stress", "lifestyle",
    "medication", "medicine", "doctor", "hospital", "symptom",
    "treatment", "prevent", "health", "wellness", "body", "weight loss",
    "gain weight", "lose weight", "fit", "fitness",
]

OUT_OF_SCOPE_RESPONSE = """## ⚠️ Out of Scope Query

I'm **MediFood AI** — a specialised health and nutrition chatbot.

I can only answer questions related to:
- 🩺 **Medical conditions** — diabetes, hypertension, heart disease, etc.
- 🥗 **Food & nutrition** — what to eat, diet plans, nutrients
- 💊 **Health recommendations** — supplements, lifestyle tips
- 📋 **Meal planning** — Indian diet plans for health conditions

**Your question appears to be outside my area of expertise.**

Please ask something like:
- *"I have diabetes, what should I eat?"*
- *"What Indian foods are good for high BP?"*
- *"Give me a weight loss diet plan"*
- *"I have anaemia, what foods have iron?"*

---
⚕️ *For medical emergencies, please contact a doctor immediately.*
"""


def is_health_query(query: str) -> bool:
    """
    Returns True if the query is health or food related.
    Returns False if it's completely off-topic.
    """
    q = query.lower()
    return any(kw in q for kw in HEALTH_KEYWORDS)
