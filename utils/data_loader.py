"""
data_loader.py
--------------
Step 1 of the RAG pipeline: Load all CSV files from the data/ folder
and convert every row into a plain-English text "chunk".

A chunk is just a sentence or two describing one record.
Example:
  "Banana is a fruit. It has 89 calories, 1.1g protein,
   23g carbs, 0.3g fat per serving."

These chunks are what gets stored in the vector database.
"""

import os


# ── helper: safe read ────────────────────────────────────────────────────────

def safe_read(path, nrows=None):
    """Try UTF-8 first, then latin-1 (handles special characters)."""
    import pandas as pd   # imported here so the chatbot runs without pandas
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(path, nrows=nrows, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, Exception):
            continue
    return pd.DataFrame()   # return empty if all fail


# ── individual loaders ───────────────────────────────────────────────────────

def load_food_nutrition(data_dir):
    """Food_Nutrition_Dataset.csv  →  food name + macros."""
    chunks = []
    df = safe_read(os.path.join(data_dir, "Food_Nutrition_Dataset.csv"))
    if df.empty:
        return chunks
    for _, row in df.iterrows():
        text = (
            f"{row.get('food_name','Unknown')} belongs to the {row.get('category','general')} "
            f"category. It contains {row.get('calories','?')} calories, "
            f"{row.get('protein','?')}g protein, {row.get('carbs','?')}g carbohydrates, "
            f"{row.get('fat','?')}g fat, {row.get('iron','?')}mg iron, "
            f"and {row.get('vitamin_c','?')}mg vitamin C per serving."
        )
        chunks.append({"text": text, "source": "Food_Nutrition_Dataset"})
    print(f"  ✓ Food_Nutrition_Dataset → {len(chunks)} chunks")
    return chunks


def load_healthy_eating(data_dir):
    """healthy_eating_dataset.csv  →  meal descriptions with diet type."""
    chunks = []
    df = safe_read(os.path.join(data_dir, "healthy_eating_dataset.csv"))
    if df.empty:
        return chunks
    for _, row in df.iterrows():
        text = (
            f"{row.get('meal_name','Unknown meal')} is a {row.get('diet_type','general')} "
            f"{row.get('meal_type','meal')} from {row.get('cuisine','unknown')} cuisine. "
            f"It has {row.get('calories','?')} kcal, {row.get('protein_g','?')}g protein, "
            f"{row.get('carbs_g','?')}g carbs, {row.get('fat_g','?')}g fat, "
            f"{row.get('fiber_g','?')}g fiber. Cooking method: {row.get('cooking_method','?')}. "
            f"Prep time: {row.get('prep_time_min','?')} mins."
        )
        chunks.append({"text": text, "source": "healthy_eating_dataset"})
    print(f"  ✓ healthy_eating_dataset → {len(chunks)} chunks")
    return chunks


def load_meal_plans(data_dir):
    """healthy_meal_plans.csv  →  meals with diet flags."""
    chunks = []
    df = safe_read(os.path.join(data_dir, "healthy_meal_plans.csv"))
    if df.empty:
        return chunks
    for _, row in df.iterrows():
        flags = []
        for flag in ["vegan", "vegetarian", "keto", "paleo", "gluten_free", "mediterranean"]:
            if str(row.get(flag, "")).strip() in ["1", "True", "true", "yes"]:
                flags.append(flag.replace("_", " "))
        flag_str = ", ".join(flags) if flags else "general"
        text = (
            f"{row.get('meal_name','Unknown')} is a healthy meal plan option "
            f"suitable for {flag_str} diets. "
            f"It has {row.get('calories','?')} kcal, {row.get('protein','?')}g protein, "
            f"{row.get('fat','?')}g fat, {row.get('carbs','?')}g carbs."
        )
        chunks.append({"text": text, "source": "healthy_meal_plans"})
    print(f"  ✓ healthy_meal_plans → {len(chunks)} chunks")
    return chunks


def load_diet_recommendations(data_dir):
    """diet_recommendations_dataset.csv  →  patient-disease-diet triples."""
    chunks = []
    df = safe_read(os.path.join(data_dir, "diet_recommendations_dataset.csv"))
    if df.empty:
        return chunks
    for _, row in df.iterrows():
        text = (
            f"A patient with {row.get('Disease_Type','unknown disease')} "
            f"(severity: {row.get('Severity','?')}) and BMI {row.get('BMI','?')} "
            f"has blood glucose {row.get('Glucose_mg/dL','?')} mg/dL and "
            f"blood pressure {row.get('Blood_Pressure_mmHg','?')} mmHg. "
            f"Dietary restrictions: {row.get('Dietary_Restrictions','none')}. "
            f"Recommended diet: {row.get('Diet_Recommendation','not specified')}."
        )
        chunks.append({"text": text, "source": "diet_recommendations_dataset"})
    print(f"  ✓ diet_recommendations_dataset → {len(chunks)} chunks")
    return chunks


def load_personalized_diet(data_dir):
    """Personalized_Diet_Recommendations.csv  →  detailed patient profiles."""
    chunks = []
    df = safe_read(os.path.join(data_dir, "Personalized_Diet_Recommendations.csv"))
    if df.empty:
        return chunks
    # Sample to keep manageable (this file can be large)
    df = df.sample(n=min(3000, len(df)), random_state=42)
    for _, row in df.iterrows():
        text = (
            f"Patient with {row.get('Chronic_Disease','no chronic disease')}, "
            f"BMI {row.get('BMI','?')}, blood sugar {row.get('Blood_Sugar_Level','?')}, "
            f"BP {row.get('Blood_Pressure_Systolic','?')}/{row.get('Blood_Pressure_Diastolic','?')} mmHg. "
            f"Allergies: {row.get('Allergies','none')}. "
            f"Recommended calories: {row.get('Recommended_Calories','?')}, "
            f"protein: {row.get('Recommended_Protein','?')}g, "
            f"carbs: {row.get('Recommended_Carbs','?')}g, "
            f"fats: {row.get('Recommended_Fats','?')}g. "
            f"Meal plan: {row.get('Recommended_Meal_Plan','not specified')}."
        )
        chunks.append({"text": text, "source": "Personalized_Diet_Recommendations"})
    print(f"  ✓ Personalized_Diet_Recommendations → {len(chunks)} chunks (sampled)")
    return chunks


def load_disease_diets(data_dir):
    """diets.csv  →  which foods suit which disease."""
    chunks = []
    df = safe_read(os.path.join(data_dir, "diets.csv"))
    if df.empty:
        return chunks
    for _, row in df.iterrows():
        diet_options = [str(row.get(f"Diet_{i}", "")).strip()
                        for i in range(1, 6) if str(row.get(f"Diet_{i}", "")).strip()]
        if diet_options:
            text = (
                f"For {row.get('Disease','unknown disease')}, "
                f"the recommended diets are: {', '.join(diet_options)}."
            )
            chunks.append({"text": text, "source": "disease_diets"})
    print(f"  ✓ diets.csv → {len(chunks)} chunks")
    return chunks


def load_precautions(data_dir):
    """precautions_df.csv  →  disease precautions."""
    chunks = []
    df = safe_read(os.path.join(data_dir, "precautions_df.csv"))
    if df.empty:
        return chunks
    for _, row in df.iterrows():
        prec = [str(row.get(f" Precaution_{i}", "")).strip()
                for i in range(1, 5) if str(row.get(f" Precaution_{i}", "")).strip() not in ["", "nan"]]
        if prec:
            text = (
                f"Precautions for {row.get('Disease','unknown disease')}: "
                f"{', '.join(prec)}."
            )
            chunks.append({"text": text, "source": "precautions"})
    print(f"  ✓ precautions_df.csv → {len(chunks)} chunks")
    return chunks


def load_workouts(data_dir):
    """workout_df.csv  →  exercise recommendations per disease."""
    chunks = []
    df = safe_read(os.path.join(data_dir, "workout_df.csv"))
    if df.empty:
        return chunks
    for _, row in df.iterrows():
        workouts = [str(row.get(f"workout_{i}", "")).strip()
                    for i in range(1, 6) if str(row.get(f"workout_{i}", "")).strip() not in ["", "nan"]]
        if workouts:
            text = (
                f"For patients with {row.get('disease','unknown disease')}, "
                f"recommended workouts/activities include: {', '.join(workouts)}."
            )
            chunks.append({"text": text, "source": "workouts"})
    print(f"  ✓ workout_df.csv → {len(chunks)} chunks")
    return chunks


def load_food_calories(data_dir):
    """Food_and_Calories.csv  →  simple calorie lookup."""
    chunks = []
    df = safe_read(os.path.join(data_dir, "Food_and_Calories.csv"))
    if df.empty:
        return chunks
    for _, row in df.iterrows():
        text = (
            f"{row.get('Food','Unknown food')} has approximately "
            f"{row.get('Calories','?')} calories per {row.get('Serving','serving')}."
        )
        chunks.append({"text": text, "source": "Food_and_Calories"})
    print(f"  ✓ Food_and_Calories.csv → {len(chunks)} chunks")
    return chunks


# ── master loader ─────────────────────────────────────────────────────────────

def load_all_data(data_dir="data"):
    """
    Calls every individual loader and returns one combined list of chunks.
    Each chunk is a dict: {"text": "...", "source": "filename"}
    """
    print("\n📂 Loading all CSV datasets...")
    all_chunks = []
    loaders = [
        load_food_nutrition,
        load_healthy_eating,
        load_meal_plans,
        load_diet_recommendations,
        load_personalized_diet,
        load_disease_diets,
        load_precautions,
        load_workouts,
        load_food_calories,
    ]
    for loader in loaders:
        try:
            chunks = loader(data_dir)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  ⚠ Skipped {loader.__name__}: {e}")

    print(f"\n✅ Total chunks loaded: {len(all_chunks)}\n")
    return all_chunks


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    chunks = load_all_data("data")
    print("Sample chunk:")
    print(chunks[0]["text"])
