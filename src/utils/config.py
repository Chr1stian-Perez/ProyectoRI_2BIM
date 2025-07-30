"""
Configuración global del sistema RAG multimodal con soporte para variables de entorno
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde archivo .env
load_dotenv()

# Configuración de rutas
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", PROJECT_ROOT / "cache"))

# Crear directorios si no existen
for dir_path in [DATA_DIR, MODELS_DIR, CACHE_DIR]:
    dir_path.mkdir(exist_ok=True)

# Configuración del modelo CLIP
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "clip-ViT-B-32")
EMBEDDING_DIMENSION = 512

# Configuración FAISS
FAISS_INDEX_TYPE = "IndexFlatIP"

# Configuración Gemini
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configuración de Kaggle
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

# Configuración OpenAI (opcional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configuración Hugging Face (opcional)
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Configuración de búsqueda
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
MIN_SIMILARITY_THRESHOLD = float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.01")) 

# Configuración de archivos soportados
SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "webp", "bmp"]
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))

# URLs de corpus reales
FLICKR8K_KAGGLE_DATASET = "adityajn105/flickr8k"
ENGLISH_DICTIONARY_KAGGLE_DATASET = "anthonytherrien/larger-dictionary-of-english-words-and-definitions"

# Configuración de corpus
FLICKR8K_MAX_IMAGES = int(os.getenv("FLICKR8K_MAX_IMAGES", "15000")) # Límite para Flickr8k
DICTIONARY_MAX_ENTRIES = int(os.getenv("DICTIONARY_MAX_ENTRIES", "80000")) # Reducido
CONCEPTS_MIN_DEFINITION_LENGTH = 10

# Configuración de logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Configuración de Kaggle API
KAGGLE_CONFIG_DIR = Path.home() / ".kaggle"
KAGGLE_CONFIG_FILE = KAGGLE_CONFIG_DIR / "kaggle.json"

def validate_environment():
    """Valida que las variables de entorno críticas estén configuradas"""
    warnings = []
    errors = []
    
    # Verificar API keys opcionales
    if not GEMINI_API_KEY:
        warnings.append("GEMINI_API_KEY no configurada - se usarán respuestas de respaldo")
    
    if not KAGGLE_USERNAME or not KAGGLE_KEY:
        if not KAGGLE_CONFIG_FILE.exists():
            warnings.append("Credenciales de Kaggle no configuradas - se usarán datos de respaldo")
    
    # Verificar directorios
    for dir_name, dir_path in [("DATA_DIR", DATA_DIR), ("MODELS_DIR", MODELS_DIR), ("CACHE_DIR", CACHE_DIR)]:
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"No se puede crear {dir_name}: {e}")
    
    return warnings, errors

def get_environment_info():
    """Retorna información sobre el entorno configurado"""
    return {
        "gemini_configured": bool(GEMINI_API_KEY),
        "kaggle_configured": bool(KAGGLE_USERNAME and KAGGLE_KEY) or KAGGLE_CONFIG_FILE.exists(),
        "openai_configured": bool(OPENAI_API_KEY),
        "huggingface_configured": bool(HUGGINGFACE_TOKEN),
        "data_dir": str(DATA_DIR),
        "models_dir": str(MODELS_DIR),
        "cache_dir": str(CACHE_DIR),
        "clip_model": CLIP_MODEL_NAME,
        "gemini_model": GEMINI_MODEL,
        "max_images": FLICKR8K_MAX_IMAGES,
        "top_k": TOP_K_RESULTS
    }
