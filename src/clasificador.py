"""
Prototipo de clasificación automática de documentos - NexaSoft Consulting

Etapas:
  1. Cargar los documentos de ejemplo
  2. Separar datos de entrenamiento y prueba
  3. Vectorizar el texto con TF-IDF
  4. Entrenar dos modelos (Naive Bayes y Regresión Logística) y compararlos
  5. Clasificar documentos nuevos en las categorías definidas
  6. Guardar resultados

Uso:  python src/clasificador.py
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from utilidades import (CARPETA_RESULTADOS, NOMBRES_CATEGORIAS, PALABRAS_VACIAS,
                        cargar_documentos, cargar_nuevos)


def linea(titulo):
    print("\n" + "=" * 70 + f"\n{titulo}\n" + "=" * 70)


# ---------------------------------------------------------------
# 1. Cargar documentos
# ---------------------------------------------------------------
linea("1. CARGA DE DOCUMENTOS")
documentos = cargar_documentos()
textos = [d["texto"] for d in documentos]
etiquetas = [d["categoria"] for d in documentos]

print(f"Documentos cargados: {len(documentos)}")
for categoria, nombre in NOMBRES_CATEGORIAS.items():
    print(f"  - {nombre:<25} {etiquetas.count(categoria)} documentos")

# ---------------------------------------------------------------
# 2. Separar entrenamiento y prueba (75% / 25%, misma proporción por categoría)
# ---------------------------------------------------------------
linea("2. SEPARACIÓN ENTRENAMIENTO / PRUEBA")
X_train, X_test, y_train, y_test, docs_train, docs_test = train_test_split(
    textos, etiquetas, documentos, test_size=0.25, stratify=etiquetas, random_state=42)
print(f"Entrenamiento: {len(X_train)} documentos | Prueba: {len(X_test)} documentos")

# ---------------------------------------------------------------
# 3 y 4. TF-IDF + modelos de clasificación
# ---------------------------------------------------------------
def crear_vectorizador():
    # strip_accents quita acentos para que "cálculo" y "calculo" sean iguales
    return TfidfVectorizer(strip_accents="unicode", lowercase=True,
                           stop_words=PALABRAS_VACIAS, ngram_range=(1, 2), sublinear_tf=True)

modelos = {
    "Naive Bayes": Pipeline([("tfidf", crear_vectorizador()), ("modelo", MultinomialNB(alpha=0.1))]),
    "Regresión Logística": Pipeline([("tfidf", crear_vectorizador()),
                                     ("modelo", LogisticRegression(max_iter=1000, C=10))]),
}

linea("3. VECTORIZACIÓN TF-IDF")
vectorizador_demo = crear_vectorizador().fit(X_train)
print(f"Tamaño del vocabulario (palabras y pares de palabras): {len(vectorizador_demo.vocabulary_)}")
print("Ejemplo de términos:", ", ".join(sorted(vectorizador_demo.vocabulary_)[:12]), "...")

linea("4. ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
resultados = {}
for nombre, pipeline in modelos.items():
    pipeline.fit(X_train, y_train)
    predicciones = pipeline.predict(X_test)
    exactitud = accuracy_score(y_test, predicciones)
    resultados[nombre] = (exactitud, predicciones)
    print(f"{nombre:<22} exactitud en prueba: {exactitud:.2%}")

# Validación cruzada de 5 partes: más confiable que una sola prueba con pocos documentos
print("\nValidación cruzada (5 partes, todos los documentos):")
for nombre, pipeline in modelos.items():
    puntajes = cross_val_score(pipeline, textos, etiquetas,
                               cv=StratifiedKFold(5, shuffle=True, random_state=42))
    print(f"{nombre:<22} exactitud promedio: {puntajes.mean():.2%} (+/- {puntajes.std():.2%})")

mejor = max(resultados, key=lambda n: resultados[n][0])
modelo_final = modelos[mejor]
predicciones = resultados[mejor][1]
print(f"\nModelo elegido: {mejor}")

print("\nResultado por documento de prueba:")
print(f"{'ID':<9}{'Categoría real':<26}{'Predicción':<26}{'¿Correcto?'}")
for doc, pred in zip(docs_test, predicciones):
    ok = "Sí" if pred == doc["categoria"] else "No"
    print(f"{doc['id']:<9}{NOMBRES_CATEGORIAS[doc['categoria']]:<26}{NOMBRES_CATEGORIAS[pred]:<26}{ok}")

print("\nReporte de clasificación:")
print(classification_report(y_test, predicciones, zero_division=0,
                            target_names=[NOMBRES_CATEGORIAS[c] for c in sorted(set(etiquetas))]))

# Matriz de confusión
CARPETA_RESULTADOS.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 6))
clases = sorted(set(etiquetas))
ConfusionMatrixDisplay.from_predictions(
    y_test, predicciones, labels=clases,
    display_labels=[NOMBRES_CATEGORIAS[c].replace(" ", "\n", 1) for c in clases],
    cmap="Blues", ax=ax, colorbar=False)
ax.set_title(f"Matriz de confusión - {mejor}")
ax.set_xlabel("Categoría predicha")
ax.set_ylabel("Categoría real")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(CARPETA_RESULTADOS / "matriz_confusion.png", dpi=150)

# ---------------------------------------------------------------
# 5. Clasificar documentos nuevos
# ---------------------------------------------------------------
linea("5. PRUEBA: CLASIFICACIÓN DE DOCUMENTOS NUEVOS")
# Se vuelve a entrenar con todos los documentos para clasificar los nuevos
modelo_final.fit(textos, etiquetas)
nuevos = cargar_nuevos()
probabilidades = modelo_final.predict_proba([n["texto"] for n in nuevos])
clases_modelo = modelo_final.classes_
for doc, prob in zip(nuevos, probabilidades):
    idx = prob.argmax()
    print(f"\n[{doc['id']}] {doc['texto'][:90]}...")
    print(f"   -> Categoría: {NOMBRES_CATEGORIAS[clases_modelo[idx]]}  (confianza {prob[idx]:.0%})")

# ---------------------------------------------------------------
# 6. Guardar el modelo
# ---------------------------------------------------------------
joblib.dump(modelo_final, CARPETA_RESULTADOS / "modelo_clasificador.joblib")
print(f"\nModelo guardado en resultados/modelo_clasificador.joblib")
print("Matriz de confusión guardada en resultados/matriz_confusion.png")
