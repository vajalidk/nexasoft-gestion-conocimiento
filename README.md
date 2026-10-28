# Prototipo de Sistema de Gestión del Conocimiento - NexaSoft Consulting

Avance de proyecto de la materia **Sistemas y tecnologías de administración del conocimiento (MAPS)**, Universidad Tecmilenio.
Autor: Vicente Alberto Jahrift Alonso López (AL07002967).

NexaSoft Consulting es una empresa **ficticia** de consultoría de TI. Este prototipo clasifica sus documentos de forma automática y permite buscarlos con preguntas en lenguaje natural.

## Estructura del repositorio

```
nexasoft-gestion-conocimiento/
├── datos/
│   ├── documentos/            42 documentos de prueba (.txt), una carpeta por categoría
│   │   ├── soporte_tecnico/
│   │   ├── desarrollo_software/
│   │   ├── infraestructura_nube/
│   │   ├── gestion_proyectos/
│   │   └── politicas_internas/
│   └── nuevos/                5 documentos sin categoría para probar el clasificador
├── src/
│   ├── utilidades.py          carga de documentos y palabras vacías en español
│   ├── clasificador.py        clasificación automática (TF-IDF + Naive Bayes / Regresión Logística)
│   └── busqueda_semantica.py  búsqueda semántica (TF-IDF + similitud coseno)
├── resultados/                salidas de ejecución, matriz de confusión y modelo guardado
├── requirements.txt
└── README.md
```

## Cómo ejecutarlo

Necesitas Python 3.9 o más nuevo.

```bash
pip install -r requirements.txt

# Clasificador
python src/clasificador.py

# Búsqueda semántica con las consultas de ejemplo
python src/busqueda_semantica.py

# Búsqueda con tu propia consulta
python src/busqueda_semantica.py "necesito la política para trabajar desde casa"

# Modo interactivo
python src/busqueda_semantica.py -i
```

## Qué hace cada script

**clasificador.py**
1. Carga los 42 documentos (la carpeta es la categoría).
2. Separa 75% para entrenar y 25% para probar.
3. Convierte el texto a vectores con TF-IDF (sin acentos, sin palabras vacías, palabras y pares de palabras).
4. Entrena Naive Bayes y Regresión Logística, los compara y se queda con el mejor.
5. Clasifica los documentos nuevos de `datos/nuevos/` y muestra la confianza.
6. Guarda la matriz de confusión y el modelo en `resultados/`.

**busqueda_semantica.py**
1. Vectoriza todos los documentos con TF-IDF.
2. Recibe una consulta en lenguaje natural.
3. Convierte la consulta a vector con el mismo vocabulario.
4. Calcula la similitud coseno contra cada documento.
5. Muestra los 3 documentos más parecidos con su puntaje.

## Categorías

| Categoría | Tipo de documentos |
| --- | --- |
| Soporte técnico | Soluciones a tickets, guías de la mesa de ayuda |
| Desarrollo de software | Documentación técnica, estándares, correcciones de bugs |
| Infraestructura y nube | Reportes de incidentes, procedimientos de operación |
| Gestión de proyectos | Lecciones aprendidas, reportes de avance, plantillas |
| Políticas internas | Políticas de la empresa y de recursos humanos |
