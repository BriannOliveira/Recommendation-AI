from flask import Flask, request, jsonify
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Carregar e pré-processar o dataset
recipes = pd.read_csv('archive/food-dataset-en.csv', nrows=5000, skiprows=[1186], on_bad_lines='skip')

# Funções auxiliares (extrair calorias, preprocessar ingredientes, calcular similaridade)
def extract_calories(energy_str):
    match = re.search(r'(\d+)\s*kcal', energy_str, re.IGNORECASE)
    return int(match.group(1)) if match else 0

def pre_processing_ingredients(recipes):
    processed_recipes = []
    for _, row in recipes.iterrows():
        items = row['ingredient'].split(", ")
        formatted_ingredients = {}
        for item in items:
            parts = item.split(": ")
            if len(parts) == 2:
                key = re.sub(r'\s+', '_', parts[0].lower().strip())
                formatted_ingredients[key] = re.sub(r'\s+', '_', parts[1].lower().strip())
        str_ingredients = ", ".join(formatted_ingredients.keys())
        calories = extract_calories(row['energy'])
        processed_recipes.append({
            'name': row['name'],
            'text': row['text'],
            'ingredients': formatted_ingredients,
            'energy': calories,
            'time_cook': row['time_cook'],
            'str_ingredients': str_ingredients
        })
    return pd.DataFrame(processed_recipes)

recipes = pre_processing_ingredients(recipes)

def calculate_similarity(ingredients_dict, str_ingredients_recipes):
    vectorizer = TfidfVectorizer()
    all_ingredients = [" ".join(ingredients_dict.keys())] + str_ingredients_recipes
    tfidf_matrix = vectorizer.fit_transform(all_ingredients)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return similarity[0]

def recommend_recipes(ingredients_list, max_kcals, num_recommendations):
    similarities = []
    for idx, row in recipes.iterrows():
        if row['energy'] <= max_kcals:
            similarity = calculate_similarity(ingredients_list, [row['str_ingredients']])
            similarities.append((row['name'], row['energy'], row['ingredients'], similarity[0]))
    similarities.sort(key=lambda x: x[3], reverse=True)
    return pd.DataFrame(similarities[:num_recommendations], columns=['nome', 'calorias', 'ingredientes', 'similaridade'])

@app.route('/recommend', methods=['POST'])
def recommend():
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json.'}), 415

    try:
        data = request.get_json()
        ingredients_list = data.get('ingredients', [])
        max_kcals = data.get('max_kcals', 500)
        num_recommendations = data.get('num_recommendations', 5)

        if not ingredients_list or not isinstance(ingredients_list, list):
            return jsonify({'error': 'Invalid or missing "ingredients" field.'}), 400

        ingredients_dict = {item['classificacao']: 1 for item in ingredients_list if 'classificacao' in item}
        recommendations = recommend_recipes(ingredients_dict, max_kcals, num_recommendations)
        return recommendations.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8082)
#         return jsonify({'error': str(e)}), 500