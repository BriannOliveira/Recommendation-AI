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
def calculate_similiraty(ingredients_dict, str_ingredients_recipes, ingredients_recipes):
    vectorizer = TfidfVectorizer()
    #print(str_ingredients_recipes)
    recipes_matrix = vectorizer.fit_transform(str_ingredients_recipes)


    targets_ingredients = " ".join(ingredients_dict.keys())
    targets_matrix = vectorizer.transform([targets_ingredients])


    similaridade = cosine_similarity(targets_matrix, recipes_matrix)

    print(similaridade)
    return similaridade[0]

# Função para recomendar receitas com base nos ingredientes e calorias fornecidas
def recomendar_receitas(lista_ingredientes, max_calorias, num_recomendacoes):
    # Lista para armazenar as similaridades
    similaridades = []

    # Itera sobre as receitas para calcular a similaridade de ingredientes e filtrar por calorias
    for idx, row in receitas.iterrows():
        if row['energy'] <= max_calorias:
            # Calcula a similaridade de ingredientes
            similaridade = calculate_similiraty(lista_ingredientes, receitas['str_ingredients'], row['ingredients'])
            similaridades.append((row['name'], row['energy'], row['ingredients'], similaridade))
            print(idx," Comparando...")

    # Ordena as receitas pela similaridade (descendente)
    similaridades.sort(key=lambda x: x[3], reverse=True)
    print("Tudo comparado!!!!!!!!!!")

    # Retorna as receitas mais semelhantes
    return pd.DataFrame(similaridades[:num_recomendacoes], columns=['nome', 'calorias', 'ingredientes', 'similaridade'])

# Exemplo de uso
lista_ingredientes = {'eggs': 2, 'parmesan': 30, 'noodle': 200}
max_calorias = 10
recomendadas = recomendar_receitas(lista_ingredientes, max_calorias, num_recomendacoes=2)
print(recomendadas)

