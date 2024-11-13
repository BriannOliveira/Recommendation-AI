import pandas as pd
import re
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# Convertendo o dataset em um DataFrame
receitas = pd.read_csv('archive/food-dataset-en.csv', skiprows=[1186], on_bad_lines='skip')

def extrair_calorias(energy_str):
    # Expressão regular para encontrar números seguidos da palavra 'kcal'
    match = re.search(r'(\d+)\s*kcal', energy_str, re.IGNORECASE)
    if match:
        # Extrai o número encontrado
        calories = int(match.group(1))
        return calories
    else:
        # Caso não encontre, retorna zero ou algum valor padrão
        return 0

def preprocessar_ingredientes(receitas):
    processed_recipes = []

    for _, row in receitas.iterrows():

        #if ":" not in row['ingredient'] or "." in row['ingredient']:
            #print(_, True)
            #continue

        #print(_, type(row['ingredient']), row['ingredient'])
        items = row['ingredient'].split(", ")
        formatted_ingredients = {}

        for item in items:
            parts = item.split(": ")
            if len(parts) == 2:
                key = re.sub(r'\s+', '_', parts[0].lower().strip())
                value = re.sub(r'\s+', '_', parts[1].lower().strip())
                formatted_ingredients[key] = value
            else:
                continue

        if not formatted_ingredients.keys():
            continue

        #print(_, formatted_ingredients.keys(),)
        str_ingredients = ", ".join(formatted_ingredients.keys())
        #print(_, str_ingredients)

        calories = extrair_calorias(row['energy'])

        processed_recipes.append({
            'name': row['name'],
            'text': row['text'],
            'ingredients': formatted_ingredients,
            'energy': calories,
            'time_cook': row['time_cook'],
            'str_ingredients': str_ingredients
        })  

    return pd.DataFrame(processed_recipes)


receitas = preprocessar_ingredientes(receitas)


# Função para calcular a similaridade entre as quantidades de ingredientes
def calculate_similiraty(ingredients_dict, str_ingredients_recipes):
    vectorizer = TfidfVectorizer()

    all_ingredients = [" ".join(ingredients_dict.keys())] + str_ingredients_recipes
    tfidf_matrix = vectorizer.fit_transform(all_ingredients)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return similarity[0]

# Função para recomendar receitas com base nos ingredientes e calorias fornecidas
def recomendar_receitas(lista_ingredientes, max_calorias, num_recomendacoes):
    # Lista para armazenar as similaridades
    similaridades = []

    # Itera sobre as receitas para calcular a similaridade de ingredientes e filtrar por calorias
    for idx, row in receitas.iterrows():
        if row['energy'] <= max_calorias:
            # Calcula a similaridade de ingredientes
            similaridade = calculate_similiraty(lista_ingredientes, receitas['str_ingredients'])
            similaridade_valor = similaridade[idx]
            similaridades.append((row['name'], row['energy'], row['ingredients'], similaridade_valor))
            print(idx," Comparando...", similaridade_valor)

    # Ordena as receitas pela similaridade (descendente)
    similaridades.sort(key=lambda x: x[3], reverse=True)
    print("Tudo comparado!!!!!!!!!!")

    # Retorna as receitas mais semelhantes
    return pd.DataFrame(similaridades[:num_recomendacoes], columns=['nome', 'calorias', 'ingredientes', 'similaridade'])

# Exemplo de uso
lista_ingredientes = {'banana': 2, 'apple': 2}
max_calorias = 100
recomendadas = recomendar_receitas(lista_ingredientes, max_calorias, num_recomendacoes=2)
print(recomendadas)

