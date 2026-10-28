"""
Funciones compartidas del prototipo de gestión del conocimiento de NexaSoft.
Aquí se cargan los documentos y se define la lista de palabras vacías en español.
"""
from pathlib import Path

# Carpeta raíz del proyecto (un nivel arriba de src/)
RAIZ = Path(__file__).resolve().parent.parent
CARPETA_DOCUMENTOS = RAIZ / "datos" / "documentos"
CARPETA_NUEVOS = RAIZ / "datos" / "nuevos"
CARPETA_RESULTADOS = RAIZ / "resultados"

# Nombres bonitos de cada categoría (la carpeta es la etiqueta)
NOMBRES_CATEGORIAS = {
    "soporte_tecnico": "Soporte técnico",
    "desarrollo_software": "Desarrollo de software",
    "infraestructura_nube": "Infraestructura y nube",
    "gestion_proyectos": "Gestión de proyectos",
    "politicas_internas": "Políticas internas",
}

# Palabras vacías en español (scikit-learn solo trae la lista en inglés)
PALABRAS_VACIAS = [
    "a", "al", "algo", "ante", "antes", "aqui", "asi", "aun", "bien", "cada", "como", "con",
    "contra", "cual", "cuando", "de", "del", "desde", "donde", "dos", "el", "ella", "ellos",
    "en", "entre", "era", "es", "esa", "ese", "eso", "esta", "estaba", "estan", "este", "esto",
    "fue", "ha", "hace", "hacia", "han", "hasta", "hay", "la", "las", "le", "les", "lo", "los",
    "mas", "me", "mi", "muy", "nada", "ni", "no", "nos", "o", "otro", "para", "pero", "poco",
    "por", "porque", "que", "se", "segun", "ser", "si", "sin", "sobre", "solo", "son", "su",
    "sus", "tambien", "tan", "te", "tiene", "tienen", "todo", "todos", "tu", "un", "una", "uno",
    "unos", "y", "ya", "yo", "debe", "deben", "puede", "pueden", "les", "despues", "luego",
]


def cargar_documentos(carpeta=CARPETA_DOCUMENTOS):
    """Lee todos los .txt de cada subcarpeta.
    Regresa una lista de diccionarios con id, categoria, titulo y texto."""
    documentos = []
    for carpeta_categoria in sorted(Path(carpeta).iterdir()):
        if not carpeta_categoria.is_dir():
            continue
        for archivo in sorted(carpeta_categoria.glob("*.txt")):
            lineas = archivo.read_text(encoding="utf-8").strip().split("\n", 1)
            documentos.append({
                "id": archivo.stem,
                "categoria": carpeta_categoria.name,
                "titulo": lineas[0],
                "texto": " ".join(lineas),  # título + contenido
            })
    return documentos


def cargar_nuevos(carpeta=CARPETA_NUEVOS):
    """Lee documentos nuevos sin categoría para probar el clasificador."""
    return [{"id": a.stem, "texto": a.read_text(encoding="utf-8").strip()}
            for a in sorted(Path(carpeta).glob("*.txt"))]
