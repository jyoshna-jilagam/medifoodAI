"""
answer_generator.py
-------------------
Step 4 of the RAG pipeline: Generate a structured, disease-specific
diet recommendation using retrieved context.

ABSOLUTE RULE — DISEASE SCOPE ENFORCEMENT:
  - Answer ONLY about the detected disease(s).
  - DO NOT mix information from other diseases.
  - Use RAG context as PRIMARY source of information.
  - Structure: 🔴 Avoid | 🟢 Recommended | 📋 Meal Plan | 💡 Tips | 💊 Supplements
  - Include specific Indian food examples.
  - Always end with doctor disclaimer.
"""


# ── Disease keyword detection ────────────────────────────────────────────────

DISEASE_KEYWORDS = {
    "Diabetes":        ["diabetes", "diabetic", "blood sugar", "glucose",
                        "insulin", "hba1c", "type 2", "type 1"],
    "Hypertension":    ["hypertension", "high blood pressure", "high bp",
                        "bp", "blood pressure"],
    "Heart Disease":   ["heart", "cardiac", "cardiovascular", "cholesterol",
                        "angina", "coronary"],
    "Obesity":         ["obesity", "obese", "overweight", "weight loss",
                        "fat loss", "bmi"],
    "Kidney Disease":  ["kidney", "renal", "nephropathy", "ckd", "creatinine"],
    "Thyroid":         ["thyroid", "hypothyroid", "hyperthyroid", "tsh"],
    "Liver Disease":   ["liver", "hepatic", "fatty liver", "jaundice", "cirrhosis"],
    "Anaemia":         ["anaemia", "anemia", "iron deficiency",
                        "haemoglobin", "hemoglobin"],
    "PCOD/PCOS":       ["pcod", "pcos", "polycystic", "hormonal imbalance"],
    "Pregnancy":       ["pregnant", "pregnancy", "prenatal", "gestational"],
    "General Health":  [],
}


# ── Indian foods to AVOID per disease ────────────────────────────────────────

INDIAN_AVOID = {
    "Diabetes": [
        "White rice (maida chawal) — switch to brown rice or millets",
        "Maida rotis, naan, puri, batura",
        "Sweets — mithai, gulab jamun, rasgulla, halwa",
        "Fruit juices and packaged sugary drinks",
        "Potatoes (aloo) in large quantity",
        "White bread and glucose biscuits",
        "Alcohol and sugary chai (limit to 1 tsp sugar max)",
    ],
    "Hypertension": [
        "Achaar (pickle) and papad — very high sodium",
        "Namkeen, chakli, bhujia — salty snacks",
        "Packaged instant noodles (Maggi etc.)",
        "Processed meats and sausages",
        "Extra cooking salt — use < 1 tsp total per day",
        "Excess ghee and butter",
        "Alcohol and excess tea or coffee",
    ],
    "Heart Disease": [
        "Vanaspati / dalda (trans fats)",
        "Red meat — mutton and beef",
        "Full-fat cream, malai, and paneer in excess",
        "Fried snacks — samosa, pakora, vada, bhajia",
        "Egg yolk more than 2 per week",
        "Coconut oil in large amounts",
        "Alcohol and smoking",
    ],
    "Obesity": [
        "Mithai and Indian sweets",
        "Fried snacks — bhujia, chakli, namkeen",
        "White rice in large portions — switch to millets",
        "Sugary lassi, shrikhand, and kulfi",
        "Maida-based items — bread, biscuits, naan",
        "Alcohol",
    ],
    "Kidney Disease": [
        "High-potassium fruits — banana, chikoo, oranges in large quantity",
        "High-phosphorus foods — cola drinks, dairy in excess",
        "Dal in large quantity if protein is restricted by doctor",
        "Pickles, papad, and salty snacks",
        "Coconut water (high potassium)",
        "Salt and salty snacks",
    ],
    "Thyroid": [
        "Raw cabbage, cauliflower, broccoli in large amounts (goitrogens when raw)",
        "Soy products in large amounts (interfere with thyroid hormone)",
        "Excess coffee and alcohol",
        "Processed foods with fluoride",
        "Refined sugar and maida",
    ],
    "Liver Disease": [
        "Alcohol — strictly zero, even one drink causes damage",
        "Fried and oily foods — samosa, pakora, puri",
        "Red meat and organ meats (liver, kidney)",
        "Excess salt and pickles",
        "Packaged and ultra-processed snacks",
        "Mithai and high-sugar items",
    ],
    "Anaemia": [
        "Tea and coffee immediately after meals — they block iron absorption",
        "Calcium-rich food at the exact same time as iron-rich food",
        "Alcohol",
        "Unsoaked phytate-rich foods in excess",
    ],
    "PCOD/PCOS": [
        "Maida-based items — white bread, naan, pasta, biscuits",
        "Sugary drinks and packaged fruit juices",
        "Mithai and high-GI Indian sweets",
        "Processed and fast food",
        "Alcohol and smoking",
        "Excess dairy (for some women with PCOS — monitor response)",
    ],
    "Pregnancy": [
        "Raw papaya and raw pineapple — can induce contractions",
        "Unpasteurised milk and soft cheeses",
        "Raw eggs and undercooked meat",
        "Excess caffeine — keep under 200 mg/day",
        "Alcohol — strictly zero throughout pregnancy",
        "Excess vitamin A supplements",
    ],
    "General Health": [
        "Fried snacks in excess",
        "Packaged processed food with preservatives",
        "Excess sugar and maida items",
        "Alcohol in excess",
    ],
}


# ── Indian foods RECOMMENDED per disease ─────────────────────────────────────

INDIAN_RECOMMEND = {
    "Diabetes": [
        "Karela (bitter gourd) juice — best taken on empty stomach, 100 ml",
        "Methi (fenugreek) seeds — soak 1 tsp overnight, eat in morning",
        "Ragi (finger millet) roti — low GI, high fibre, very filling",
        "Brown rice or millets (jowar, bajra) instead of white rice",
        "Moong dal and masoor dal — high protein, low GI",
        "Lauki (bottle gourd) and palak (spinach) sabzi daily",
        "Amla (Indian gooseberry) — raw or juice helps regulate sugar",
        "Jamun (Indian blackberry) — traditional diabetes management fruit",
        "Chaas (buttermilk, plain, no salt) — excellent probiotic",
        "5–6 soaked almonds and 2 walnuts daily",
    ],
    "Hypertension": [
        "Lauki (bottle gourd) juice — 100 ml morning, natural BP reducer",
        "2 raw garlic (lahsun) cloves daily on empty stomach",
        "Banana — rich in potassium to counteract sodium effects",
        "Palak dal — high potassium and magnesium",
        "Fresh coconut water (not packaged) — natural electrolyte",
        "Amla juice — antioxidant, supports blood pressure management",
        "Moong dal khichdi with low salt — easily digestible",
        "Oats dalia for breakfast — soluble fibre lowers BP naturally",
        "Watermelon — contains citrulline which relaxes blood vessels",
        "Coriander (dhania) seed water — mild natural diuretic",
    ],
    "Heart Disease": [
        "Walnuts (akhrot) — 7–8 daily, rich in omega-3",
        "Flaxseeds (alsi) — 1 tbsp ground added to roti dough or chaas",
        "Oats upma or dalia — beta-glucan lowers LDL cholesterol",
        "Garlic tadka in dal — cardioprotective compound allicin",
        "Turmeric milk (haldi doodh) — anti-inflammatory, no sugar",
        "Pomegranate (anaar) juice — strong antioxidant for heart",
        "Palak (spinach) sabzi — folate and natural nitrates",
        "Fish — rohu, katla, hilsa — omega-3 fatty acids",
        "Rajma and chhole (1–2 times/week) — plant protein, fibre",
        "Mustard oil or olive oil for cooking (not coconut or palm)",
    ],
    "Obesity": [
        "Moong dal soup without ghee — high protein, very filling",
        "Cucumber raita with jeera — low calorie, cooling",
        "Jowar or bajra roti — more fibre and nutrients than wheat",
        "Roasted chana (unsalted) — high protein snack",
        "Green tea without sugar — mild metabolism support",
        "Lauki (bottle gourd) sabzi — very low calorie vegetable",
        "Sprouts salad (moong, chana) — protein and fibre together",
        "Low-fat curd with jeera — probiotic support",
        "Warm lemon water in morning — hydration and digestive support",
        "Ragi roti or ragi mudde — keeps you full for hours",
    ],
    "Kidney Disease": [
        "Cabbage (patta gobhi) sabzi — low potassium, kidney-friendly",
        "White rice (not brown) — lower phosphorus than brown rice",
        "Egg whites only — high quality protein, low phosphorus",
        "Apple and berries — lower-potassium fruits",
        "Cauliflower sabzi — kidney-friendly vegetable",
        "Arbi (colocasia) — boil and discard water to reduce potassium",
        "Sago (sabudana) khichdi — low protein, easy on kidneys",
    ],
    "Thyroid": [
        "2 Brazil nuts daily — excellent selenium for thyroid hormone conversion",
        "Iodised salt — use in moderate quantity only",
        "Eggs (poached or boiled) — selenium and iodine source",
        "Fish (rohu, bangda, surmai) — natural iodine",
        "Rajma and moong dal — zinc supports thyroid function",
        "Cooked leafy greens — cooking reduces goitrogens significantly",
        "Pumpkin seeds — zinc and magnesium for thyroid health",
    ],
    "Liver Disease": [
        "Amla juice — most powerful liver antioxidant available in India",
        "Turmeric (haldi) in every dal and sabzi — curcumin protects liver cells",
        "Green tea without sugar — catechins support liver detox",
        "Methi (fenugreek) seed water — traditional liver support",
        "Moong dal — very light on the liver, easy to digest",
        "Lauki and tinda sabzi — gentle vegetables for liver patients",
        "Ripe papaya — enzyme papain aids digestion",
        "Beetroot (chukandar) salad — betalains protect liver",
    ],
    "Anaemia": [
        "Palak (spinach) dal with lemon — iron + vitamin C combination",
        "Bajra roti — one of the highest plant iron sources",
        "Pomegranate (anaar) daily — traditional haemoglobin builder",
        "5–6 dates (khajoor) daily — iron and natural energy",
        "Til (sesame) chutney or til laddoo — very high iron content",
        "Kala chana (black chickpea) curry — good plant iron",
        "Always add lemon or amla alongside iron-rich meals",
        "Beetroot (chukandar) juice — popular traditional iron remedy",
        "Jaggery (gud) in small quantities — iron source",
    ],
    "PCOD/PCOS": [
        "Ragi roti — low GI, high calcium, excellent for hormonal health",
        "Methi seed water in morning — helps regulate insulin resistance",
        "Flaxseeds (alsi) in roti dough — phytoestrogens help balance oestrogen",
        "Moong and masoor dal — plant protein, keeps blood sugar stable",
        "Sabja (basil seeds) in water — anti-inflammatory",
        "Turmeric (haldi) milk — powerful anti-inflammatory drink",
        "Whole fruits not juice — fibre keeps glycaemic index low",
        "Pumpkin seeds — zinc supports ovarian health",
    ],
    "Pregnancy": [
        "Ragi porridge — calcium, iron, and protein in one bowl",
        "Moong dal khichdi — complete protein, gentle on digestion",
        "Palak paneer — iron and calcium together",
        "Banana — potassium, folate, and quick energy",
        "Sesame chikki (til) — calcium and iron",
        "Dahi (curd) — probiotics and calcium",
        "Dates (khajoor) — iron and natural energy boost",
        "Fresh coconut water — hydration and electrolytes",
        "Amla — vitamin C and immunity support",
    ],
    "General Health": [
        "Dal-chawal with salad — balanced Indian thali",
        "Fresh seasonal fruits daily",
        "Curd or chaas with meals",
        "Seasonal sabzi with minimal oil",
        "Handful of mixed nuts daily",
    ],
}


# ── Sample Indian meal plans per disease ─────────────────────────────────────

MEAL_PLANS = {
    "Diabetes": {
        "⏰ Early morning (6–7 AM)": "1 glass methi seed water (1 tsp soaked overnight) or 30 ml amla juice. 5–6 soaked almonds.",
        "🍳 Breakfast (8–9 AM)": "2 ragi or jowar rotis with palak/lauki sabzi OR oats upma with vegetables. Chai with 1 tsp sugar max, no biscuits.",
        "🥗 Mid-morning (11 AM)": "1 small bowl sprouts chaat OR 1 guava or 10 jamuns. NO banana, mango, or chikoo.",
        "🍛 Lunch (1–2 PM)": "2 multigrain rotis + 1 katori moong or masoor dal + 1 katori sabzi + 1 katori plain curd (no sugar) + salad. Brown rice (½ cup max) on alternate days only.",
        "☕ Evening (4–5 PM)": "Handful roasted chana OR cucumber slices with unsweetened chaas.",
        "🌙 Dinner (7–8 PM)": "2 rotis + 1 katori dal or egg curry + 1 light vegetable sabzi. Dinner should be the lightest meal of your day.",
        "🌛 Bedtime (10 PM)": "1 glass warm haldi (turmeric) milk with no sugar — helps manage fasting blood sugar.",
    },
    "Hypertension": {
        "⏰ Early morning (6–7 AM)": "2 raw garlic cloves with warm water OR 100 ml lauki juice (no added salt).",
        "🍳 Breakfast (8–9 AM)": "Oats dalia with vegetables (no salt added) + 1 banana OR Poha with very minimal salt and peanuts.",
        "🥗 Mid-morning (11 AM)": "Fresh coconut water OR 1 seasonal fruit.",
        "🍛 Lunch (1–2 PM)": "2 rotis (prepared without salt in dough) + dal (light tadka, minimal salt) + seasonal sabzi + 1 katori low-fat curd.",
        "☕ Evening (4–5 PM)": "Unsalted roasted makhana (foxnuts) OR a fresh fruit.",
        "🌙 Dinner (7–8 PM)": "Moong dal khichdi with very low salt + steamed vegetables OR 2 rotis + light vegetable sabzi.",
        "🌛 Bedtime (10 PM)": "1 glass warm low-fat milk (no sugar).",
    },
    "Heart Disease": {
        "⏰ Early morning (6–7 AM)": "8–10 soaked walnuts + 1 tsp ground flaxseeds stirred in water OR 100 ml pomegranate juice.",
        "🍳 Breakfast (8–9 AM)": "Oats upma with mixed vegetables + 1 cup green tea (no sugar).",
        "🥗 Mid-morning (11 AM)": "1 fruit (apple, pear, or orange) + a small handful of almonds.",
        "🍛 Lunch (1–2 PM)": "2 rotis + rajma or chhole (once a week max) OR fish curry (rohu/katla) + palak dal + large salad.",
        "☕ Evening (4–5 PM)": "1 cup green tea + roasted makhana or unsalted chana.",
        "🌙 Dinner (7–8 PM)": "2 rotis + moong dal (light) + steamed or lightly stir-fried vegetables in 1 tsp mustard oil.",
        "🌛 Bedtime (10 PM)": "Haldi milk with low-fat milk, no sugar.",
    },
    "Obesity": {
        "⏰ Early morning (6–7 AM)": "1 glass warm lemon water (no sugar). 5 soaked almonds.",
        "🍳 Breakfast (8–9 AM)": "2 moong dal chillas cooked on non-stick pan (minimal oil) + 1 bowl low-fat curd OR vegetable oats upma (no ghee).",
        "🥗 Mid-morning (11 AM)": "1 fruit (apple, pear, guava — avoid banana and mango) OR cucumber + tomato salad with lime.",
        "🍛 Lunch (1–2 PM)": "2 jowar or bajra rotis + 1 katori dal + large portion sabzi + salad. Avoid rice completely.",
        "☕ Evening (4–5 PM)": "1 cup green tea + small handful roasted chana (no salt) OR sprouted moong.",
        "🌙 Dinner (7–8 PM)": "Large bowl vegetable soup OR 1 roti + dal + sabzi. Keep dinner under 400 calories.",
        "🌛 Bedtime": "Do not eat after 8 PM. Ajwain water or plain warm water if hungry.",
    },
    "Anaemia": {
        "⏰ Early morning (6–7 AM)": "1 glass pomegranate juice OR beetroot-amla juice (powerful iron + vitamin C combination).",
        "🍳 Breakfast (8–9 AM)": "2 bajra rotis with palak sabzi + 1 amla OR Poha with a generous lemon squeeze (vitamin C boosts iron absorption).",
        "🥗 Mid-morning (11 AM)": "5–6 dates (khajoor) OR a piece of til (sesame) chikki.",
        "🍛 Lunch (1–2 PM)": "2 rotis + kala chana curry + palak paneer + salad with lemon dressing. The lemon is critical for iron absorption.",
        "☕ Evening (4–5 PM)": "Sprouted moong chaat OR roasted chana. NO tea or coffee within 1 hour of iron-rich meals.",
        "🌙 Dinner (7–8 PM)": "2 bajra or ragi rotis + moong dal + green vegetable sabzi with lemon.",
        "🌛 Bedtime": "1 glass warm milk with a pinch of turmeric.",
    },
    "General Health": {
        "⏰ Early morning": "Warm water with lemon + 5 soaked almonds.",
        "🍳 Breakfast": "Dal paratha or oats with fruit.",
        "🥗 Mid-morning": "Seasonal fruit or buttermilk.",
        "🍛 Lunch": "2 rotis + dal + sabzi + curd + salad — a balanced thali.",
        "☕ Evening": "Chai (1 tsp sugar) + roasted chana or makhana.",
        "🌙 Dinner": "Lighter than lunch — khichdi, dal roti, or vegetable soup.",
    },
}


# ── Supplements per disease ───────────────────────────────────────────────────

SUPPLEMENTS = {
    "Diabetes": [
        "Vitamin D — most diabetics are deficient, get levels tested",
        "Magnesium — from almonds, seeds, or supplement (discuss with doctor)",
        "Chromium — helps insulin sensitivity (doctor prescription)",
        "Berberine — natural glucose-lowering, discuss dosage with doctor",
    ],
    "Hypertension": [
        "Magnesium glycinate 400 mg — if deficient (get tested)",
        "Potassium — from food first (banana, lauki), supplements only if prescribed",
        "CoQ10 — cardioprotective, discuss with cardiologist",
        "Vitamin D — deficiency raises BP risk",
    ],
    "Heart Disease": [
        "Omega-3 fish oil 1–2 g EPA+DHA — discuss with cardiologist",
        "CoQ10 100–200 mg — especially if on statin medication",
        "Vitamin D — deficiency linked to heart disease risk",
        "Magnesium — supports normal heart rhythm",
    ],
    "Obesity": [
        "Vitamin D — most people with obesity have severe deficiency",
        "Vitamin B12 — especially if vegetarian/vegan",
        "Iron — get levels checked before supplementing",
        "Protein powder — only if dietary protein intake is genuinely insufficient",
    ],
    "Anaemia": [
        "Iron supplement as prescribed by doctor (ferrous sulfate or bisglycinate)",
        "Vitamin C 500 mg alongside every iron dose — doubles absorption",
        "Vitamin B12 — check levels, very common deficiency",
        "Folic acid — essential for red blood cell production",
    ],
    "PCOD/PCOS": [
        "Myo-inositol 2–4 g/day — discuss with gynaecologist",
        "Vitamin D — very commonly deficient in PCOS",
        "Zinc 30 mg — supports hormonal balance",
        "Omega-3 fish oil — reduces ovarian inflammation",
    ],
    "Thyroid": [
        "Selenium 200 mcg — critical for T4 to T3 conversion (doctor guidance)",
        "Zinc 30 mg — supports thyroid hormone production",
        "Vitamin D — hypothyroidism is associated with deficiency",
        "Iodine — ONLY supplement if deficiency confirmed by test",
    ],
    "Pregnancy": [
        "Iron + folic acid as prescribed by obstetrician",
        "Calcium 500 mg twice daily from second trimester",
        "Vitamin D — discuss dosage with your doctor",
        "Omega-3 DHA 200–300 mg — for baby brain development",
    ],
    "Kidney Disease": [
        "IMPORTANT: Do NOT take iron, potassium, or phosphorus supplements without nephrologist approval",
        "Follow your nephrologist's exact supplement protocol",
        "Vitamin D (activated form) may be prescribed — only under supervision",
    ],
    "Liver Disease": [
        "Milk thistle (silymarin) 140–420 mg — discuss with hepatologist",
        "Vitamin E — antioxidant, discuss dosage with doctor",
        "B-complex vitamins — liver disease depletes B vitamins",
        "Zinc — often deficient in liver disease",
    ],
    "General Health": [
        "Vitamin D 1000–2000 IU — most Indians are deficient",
        "Vitamin B12 — especially for vegetarians",
        "Iron — get Hb tested before supplementing",
        "Omega-3 — if fish intake is low",
    ],
}


# ── Lifestyle tips per disease ────────────────────────────────────────────────

LIFESTYLE_TIPS = {
    "Diabetes": [
        "Walk 30 minutes after each main meal — this alone reduces post-meal sugar spike significantly",
        "Eat every 3–4 hours — never skip meals, especially breakfast",
        "Monitor fasting sugar and 2-hour post-meal sugar at home",
        "Drink 8–10 glasses of water — dehydration raises blood sugar",
        "Sleep 7–8 hours — poor sleep raises insulin resistance by 25%",
        "Manage stress with pranayama daily — cortisol directly raises blood glucose",
        "Use smaller plates — portion control is as important as food choice",
    ],
    "Hypertension": [
        "Check BP at the same time every day (morning, before eating or medication)",
        "Reduce total salt to under 1 tsp per day including all cooking salt",
        "Walk 45 minutes daily or swim — proven to lower systolic BP by 5–8 points",
        "Quit smoking completely — each cigarette spikes BP immediately for 30 minutes",
        "Practice anulom vilom pranayama 10 minutes daily — reduces BP naturally",
        "Every 5 kg of weight loss reduces systolic BP by approximately 5 mmHg",
    ],
    "Heart Disease": [
        "Cardiac rehab walking: start 15 min/day and build to 45 min over 4–6 weeks",
        "Monitor LDL, HDL, and triglycerides every 3 months",
        "Never skip prescribed cardiac medications without doctor approval",
        "Avoid intense isometric exercise (heavy weight lifting) without cardiologist clearance",
        "Manage stress — emotional triggers are a leading cause of cardiac events",
        "Sleep on your left side if recommended by your doctor",
    ],
    "Obesity": [
        "Aim for 500-calorie deficit daily — this equals approximately 0.5 kg loss per week",
        "Eat slowly and chew thoroughly — the brain takes 20 minutes to register fullness",
        "Strength training 3 times a week preserves muscle mass during fat loss",
        "Weigh yourself once a week at the same time — not daily, weight fluctuates naturally",
        "Drink a full glass of water before every meal — reduces calorie intake by 15–20%",
        "Sleep 7–8 hours — poor sleep increases hunger hormones ghrelin and leptin",
    ],
    "Anaemia": [
        "Cook in a traditional iron kadai (iron wok) — leaches beneficial iron into food",
        "Drink amla juice daily — the vitamin C dramatically boosts iron absorption from food",
        "Never drink tea or coffee within 1 hour before or after iron-rich meals",
        "Get Hb levels checked every 3 months during anaemia treatment",
        "Pair iron-rich foods with vitamin C at every meal: dal with lemon, palak with tomato",
    ],
    "General Health": [
        "Sleep 7–8 hours consistently every night",
        "Exercise 30 minutes daily — walking counts",
        "Eat 5 portions of vegetables and fruits daily",
        "Drink 8 glasses of water throughout the day",
        "Manage stress with yoga, walking, or breathing exercises",
    ],
}


# ── Disease detection ─────────────────────────────────────────────────────────

def detect_diseases(query):
    """Find which diseases the user mentioned in their query."""
    q = query.lower()
    found = []
    for disease, keywords in DISEASE_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            found.append(disease)
    if len(found) > 1 and "General Health" in found:
        found.remove("General Health")
    return found if found else ["General Health"]


def get_rag_context(retrieved_chunks, disease_label):
    """
    Filter retrieved chunks to only include those relevant to
    the detected disease — enforcing the disease scope rule.
    """
    disease_lower = disease_label.lower().replace(" disease", "").split("/")[0].strip()

    priority_sources = [
        "disease_diets", "diet_recommendations_dataset",
        "Personalized_Diet_Recommendations", "healthy_meal_plans",
        "healthy_eating_dataset",
    ]

    lines = []
    for chunk in retrieved_chunks:
        text_lower = chunk["text"].lower()
        if disease_lower in text_lower or chunk["source"] in priority_sources:
            lines.append(f"• [{chunk['source']}] {chunk['text']}")

    if not lines:
        for chunk in retrieved_chunks[:5]:
            lines.append(f"• [{chunk['source']}] {chunk['text']}")

    return "\n".join(lines[:6])


# ── Main answer generator ─────────────────────────────────────────────────────

def generate_answer(query, retrieved_chunks, patient_profile=None):
    """
    Build a structured, disease-specific diet answer.

    ABSOLUTE RULE: Answer ONLY about the detected disease(s).
    DO NOT mix information from other diseases.
    Use RAG context as PRIMARY source of information.

    Parameters
    ----------
    query            : str  – user's question
    retrieved_chunks : list – from retriever.retrieve_top_k()
    patient_profile  : dict – optional e.g. {age, gender, bmi, conditions}

    Returns
    -------
    answer : str – formatted markdown diet plan
    """
    diseases      = detect_diseases(query)
    disease_label = " + ".join(diseases)
    primary       = diseases[0]

    # ── RAG context (PRIMARY source) ────────────────────────────────────────
    rag_context = get_rag_context(retrieved_chunks, disease_label)

    # ── Patient profile ─────────────────────────────────────────────────────
    if patient_profile and any(v for v in patient_profile.values()):
        profile_lines = [f"- {k}: {v}" for k, v in patient_profile.items() if v]
        profile_str = "\n".join(profile_lines)
    else:
        profile_str = "- Profile not provided (showing general recommendations for this condition)"

    # ── DISEASE SCOPE: Pull data ONLY for detected disease(s) ───────────────
    avoid_list  = INDIAN_AVOID.get(primary,   INDIAN_AVOID["General Health"])
    rec_list    = INDIAN_RECOMMEND.get(primary, INDIAN_RECOMMEND["General Health"])
    meal_plan   = MEAL_PLANS.get(primary,     MEAL_PLANS["General Health"])
    supplements = SUPPLEMENTS.get(primary,    SUPPLEMENTS["General Health"])
    tips        = LIFESTYLE_TIPS.get(primary, LIFESTYLE_TIPS["General Health"])

    # For combined conditions (e.g. diabetes + hypertension), add extra items
    extra_avoid = []
    extra_rec   = []
    for d in diseases[1:]:
        extra_avoid.extend(INDIAN_AVOID.get(d, [])[:3])
        extra_rec.extend(INDIAN_RECOMMEND.get(d, [])[:3])

    # ── Build response ───────────────────────────────────────────────────────
    answer = f"""## 🩺 MediFood AI — Personalised Diet Plan

**Condition(s):** {disease_label}

**Patient profile:**
{profile_str}

---

### 📋 What your health knowledge base says
{rag_context if rag_context.strip() else "*(Showing curated recommendations for this condition)*"}

---

### 🔴 Foods to Strictly Avoid
"""
    for item in avoid_list:
        answer += f"- {item}\n"
    if extra_avoid:
        answer += "\n**Also avoid (combined conditions):**\n"
        for item in extra_avoid:
            answer += f"- {item}\n"

    answer += f"""
---

### 🟢 Recommended Foods for {disease_label}
"""
    for item in rec_list:
        answer += f"- {item}\n"
    if extra_rec:
        answer += "\n**Also beneficial for your combined conditions:**\n"
        for item in extra_rec:
            answer += f"- {item}\n"

    answer += f"""
---

### 📋 Sample Daily Indian Meal Plan
"""
    for meal_time, meal_desc in meal_plan.items():
        answer += f"**{meal_time}**\n{meal_desc}\n\n"

    answer += "---\n\n### 💡 Lifestyle Tips\n"
    for tip in tips:
        answer += f"- {tip}\n"

    answer += "\n---\n\n### 💊 Supplements (Discuss with Your Doctor First)\n"
    for supp in supplements:
        answer += f"- {supp}\n"

    answer += """
---

⚕️ *Always consult your doctor for personalised medical advice specific to your test reports and medications.*
"""
    return answer.strip()
