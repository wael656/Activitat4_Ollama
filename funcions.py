import ollama 
import numpy as np
from numpy.linalg import norm

MODELO = "llama3.2"

def cargar_preguntes():
    llista_preguntes =[]
    print ("Carregant...")

    try: 
        with open('preguntes.txt', 'r', encoding='utf-8') as fitxer:
            for linia in fitxer:
                if "|" in linia:
                    partes = linia.split('|')
                    pregunta = partes[0].strip()
                    respuesta = partes[1].strip()


                    response = ollama.embeddings(model=MODELO, prompt=pregunta)
                    embedding = response['embedding']

                    llista_preguntes.append({
                        'questio': pregunta,
                        'resposta': respuesta,
                        'embedding': embedding
                    })
        return llista_preguntes
    except FileNotFoundError:
        print("Error: no es troba el fitxer de preguntes")
        return[]

def calcular_similitud(vec1, vec2):
    A = np.array(vec1)
    B = np.array(vec2)
    
    cosine = np.dot(A, B) / (norm(A) * norm(B))
    return cosine