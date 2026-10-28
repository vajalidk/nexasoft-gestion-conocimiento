"""
Búsqueda semántica básica - NexaSoft Consulting

Etapas:
  1. Cargar los documentos y vectorizarlos con TF-IDF
  2. Recibir una consulta en lenguaje natural
  3. Convertir la consulta a vector TF-IDF (con el mismo vocabulario)
  4. Calcular la similitud coseno entre la consulta y cada documento
  5. Mostrar los documentos más parecidos

Uso:
  python src/busqueda_semantica.py                      -> corre las consultas de ejemplo
  python src/busqueda_semantica.py "mi pregunta aquí"   -> busca tu propia consulta
  python src/busqueda_semantica.py -i                   -> modo interactivo
"""
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utilidades import NOMBRES_CATEGORIAS, PALABRAS_VACIAS, cargar_documentos

# ---------------------------------------------------------------
# 1. Cargar y vectorizar los documentos
# ---------------------------------------------------------------
documentos = cargar_documentos()
vectorizador = TfidfVectorizer(strip_accents="unicode", lowercase=True,
                               stop_words=PALABRAS_VACIAS, sublinear_tf=True)
matriz_documentos = vectorizador.fit_transform([d["texto"] for d in documentos])


def buscar(consulta, top=3):
    """Pasos 2 a 5: recibe la consulta, la vectoriza, calcula la similitud y ordena."""
    vector_consulta = vectorizador.transform([consulta])                       # paso 3
    similitudes = cosine_similarity(vector_consulta, matriz_documentos)[0]     # paso 4
    orden = similitudes.argsort()[::-1][:top]                                  # paso 5
    return [(documentos[i], similitudes[i]) for i in orden if similitudes[i] > 0]


def mostrar(consulta, top=3):
    print("\n" + "=" * 70)
    print(f'CONSULTA: "{consulta}"')
    print("=" * 70)
    resultados = buscar(consulta, top)
    if not resultados:
        print("No encontré documentos relacionados.")
        return
    for pos, (doc, score) in enumerate(resultados, 1):
        print(f"{pos}. [{doc['id']}] {doc['titulo']}")
        print(f"   Categoría: {NOMBRES_CATEGORIAS[doc['categoria']]} | Similitud: {score:.3f}")


CONSULTAS_EJEMPLO = [
    "el usuario no puede entrar porque su cuenta está bloqueada",
    "qué hacer si se cae el servidor por falta de memoria",
    "cómo estimar el tiempo y costo de un proyecto nuevo",
    "los pedidos de la tienda no llegan a SAP",
]

if __name__ == "__main__":
    print(f"Documentos indexados: {len(documentos)} | Términos en el vocabulario: {len(vectorizador.vocabulary_)}")
    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        while True:
            consulta = input("\nEscribe tu consulta (o 'salir'): ").strip()
            if consulta.lower() == "salir":
                break
            mostrar(consulta)
    elif len(sys.argv) > 1:
        mostrar(" ".join(sys.argv[1:]))
    else:
        for c in CONSULTAS_EJEMPLO:
            mostrar(c)
