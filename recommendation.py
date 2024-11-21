import os
import pandas as pd
import re
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# URL para obter imagens temporárias e classificá-las no backend da IA
IA_BASE_URL = "https://divine-moving-yeti.ngrok-free.app"
IMAGES_DIR_URL = f"{IA_BASE_URL}/temp_images"  # URL para o diretório de imagens temporárias
CLASSIFICAR_URL = f"{IA_BASE_URL}/classificar"  # URL para classificar a imagem

# Função para obter lista de imagens temporárias
def obter_imagens_temp():
    response = requests.get(IMAGES_DIR_URL)
    if response.status_code == 200:
        try:
            images = response.json()  # Supondo que a resposta seja uma lista de URLs das imagens
            return images
        except ValueError:
            print("Erro ao decodificar a resposta JSON.")
            return []
    else:
        print(f"Erro ao buscar imagens temporárias! Status: {response.status_code}")
        return []

# Função para enviar imagem e obter classificações do backend de IA
def buscar_ingredientes_da_ia(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        files = {'file': (os.path.basename(image_url), response.content, 'image/jpeg')}
        response = requests.post(CLASSIFICAR_URL, files=files)

        if response.status_code == 200:
            print("Dados recebidos com sucesso!")
            try:
                data = response.json()  # Dados retornados da IA contendo os ingredientes
                return data
            except ValueError:
                print("Erro ao decodificar a resposta JSON.")
                return None
        else:
            print(f"Erro ao buscar os dados! Status: {response.status_code}, {response.text}")
            return None
    else:
        print(f"Erro ao baixar imagem temporária! Status: {response.status_code}")
        return None

# Carregar e pré-processar o dataset
receitas = pd.read_csv('archive/food-dataset-en.csv', nrows=5000, skiprows=[1186], on_bad_lines='skip')

# Funções auxiliares (extrair calorias, preprocessar ingredientes, calcular similaridade)
def extrair_calorias(energy_str):
    match = re.search(r'(\d+)\s*kcal', energy_str, re.IGNORECASE)
    return int(match.group(1)) if match else 0

def preprocessar_ingredientes(receitas):
    # Processamento dos ingredientes
    processed_recipes = []
    for _, row in receitas.iterrows():
        items = row['ingredient'].split(", ")
        formatted_ingredients = {}
        for item in items:
            parts = item.split(": ")
            if len(parts) == 2:
                key = re.sub(r'\s+', '_', parts[0].lower().strip())
                formatted_ingredients[key] = re.sub(r'\s+', '_', parts[1].lower().strip())
        str_ingredients = ", ".join(formatted_ingredients.keys())
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

def calculate_similarity(ingredients_dict, str_ingredients_recipes):
    vectorizer = TfidfVectorizer()
    all_ingredients = [" ".join(ingredients_dict.keys())] + str_ingredients_recipes
    tfidf_matrix = vectorizer.fit_transform(all_ingredients)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return similarity[0]

# Função para recomendar receitas com base nos ingredientes e calorias
def recomendar_receitas(lista_ingredientes, max_calorias, num_recomendacoes):
    similaridades = []
    for idx, row in receitas.iterrows():
        if row['energy'] <= max_calorias:
            # Aqui calculamos a similaridade para cada receita com os ingredientes da IA
            similaridade = calculate_similarity(lista_ingredientes, [row['str_ingredients']])
            similaridade_valor = similaridade[0]
            similaridades.append((row['name'], row['energy'], row['ingredients'], similaridade_valor))
    similaridades.sort(key=lambda x: x[3], reverse=True)
    return pd.DataFrame(similaridades[:num_recomendacoes], columns=['nome', 'calorias', 'ingredientes', 'similaridade'])

# Função principal para buscar imagens temporárias, obter ingredientes e recomendar receitas para cada imagem
def main():
    imagens_temp = obter_imagens_temp()
    if not imagens_temp:
        print("Nenhuma imagem temporária encontrada.")
        return

    for image_url in imagens_temp:
        ingredientes_da_ia = buscar_ingredientes_da_ia(image_url)
        if ingredientes_da_ia and 'predictions' in ingredientes_da_ia:
            # Aqui extraímos os ingredientes da IA e convertemos para o formato necessário
            lista_ingredientes = {item['classificacao']: 1 for item in ingredientes_da_ia['predictions']}
            max_calorias = 100  # Exemplo de limite de calorias
            recomendadas = recomendar_receitas(lista_ingredientes, max_calorias, num_recomendacoes=2)
            print(f"\nRecomendações para a imagem {image_url}:")
            print(recomendadas)
        else:
            print(f"Dados de ingredientes não encontrados para a imagem {image_url}")

# Executa a função principal
if __name__ == "__main__":
    main()
