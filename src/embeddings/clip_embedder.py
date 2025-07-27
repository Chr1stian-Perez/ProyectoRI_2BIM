"""
Generador de embeddings multimodales usando CLIP
"""
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
import hashlib
import pickle
from pathlib import Path
from typing import List, Union, Optional
import requests
from io import BytesIO
import sys

# Configurar path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# Importaciones absolutas
try:
    from src.utils.config import CLIP_MODEL_NAME, EMBEDDING_DIMENSION, CACHE_DIR
    from src.utils.logger import setup_logger
except ImportError:
    from utils.config import CLIP_MODEL_NAME, EMBEDDING_DIMENSION, CACHE_DIR
    from utils.logger import setup_logger

logger = setup_logger(__name__)

class CLIPEmbedder:
    """Generador de embeddings multimodales usando CLIP"""
    
    def __init__(self):
        self.model = None
        self.cache_dir = CACHE_DIR / "embeddings"
        self.cache_dir.mkdir(exist_ok=True)
        
    def _load_model(self):
        """Lazy loading del modelo CLIP"""
        if self.model is None:
            logger.info(f"Cargando modelo CLIP: {CLIP_MODEL_NAME}")
            self.model = SentenceTransformer(CLIP_MODEL_NAME)
            logger.info("Modelo CLIP cargado exitosamente")
    
    def _get_cache_key(self, content: Union[str, bytes]) -> str:
        """Genera clave de caché usando hash MD5"""
        if isinstance(content, str):
            content = content.encode('utf-8')
        return hashlib.md5(content).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Carga embedding desde caché"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error cargando caché {cache_key}: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """Guarda embedding en caché"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Error guardando caché {cache_key}: {e}")
    
    def encode_text(self, texts: Union[str, List[str]], use_cache: bool = True) -> np.ndarray:
        """
        Genera embeddings para texto(s)
        
        Args:
            texts: Texto o lista de textos
            use_cache: Si usar caché para embeddings
            
        Returns:
            Array numpy con embeddings normalizados
        """
        self._load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        for text in texts:
            cache_key = self._get_cache_key(text) if use_cache else None
            
            if use_cache and cache_key:
                cached_embedding = self._load_from_cache(cache_key)
                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                    continue
            
            # Generar embedding
            embedding = self.model.encode([text])[0]
            embedding = embedding / np.linalg.norm(embedding)  # Normalizar
            
            if use_cache and cache_key:
                self._save_to_cache(cache_key, embedding)
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def encode_image(self, image: Union[Image.Image, str], use_cache: bool = True) -> np.ndarray:
        """
        Genera embedding para imagen
        
        Args:
            image: Imagen PIL o URL/path de imagen
            use_cache: Si usar caché para embeddings
            
        Returns:
            Array numpy con embedding normalizado
        """
        self._load_model()
        
        # Procesar imagen
        if isinstance(image, str):
            if image.startswith('http'):
                # URL de imagen
                response = requests.get(image)
                pil_image = Image.open(BytesIO(response.content))
            else:
                # Path local
                pil_image = Image.open(image)
        else:
            pil_image = image
        
        # Convertir a RGB si es necesario
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Generar clave de caché
        cache_key = None
        if use_cache:
            # Para imágenes, usamos un hash del contenido
            img_bytes = BytesIO()
            pil_image.save(img_bytes, format='PNG')
            cache_key = self._get_cache_key(img_bytes.getvalue())
            
            cached_embedding = self._load_from_cache(cache_key)
            if cached_embedding is not None:
                return cached_embedding
        
        # Generar embedding
        embedding = self.model.encode([pil_image])[0]
        embedding = embedding / np.linalg.norm(embedding)  # Normalizar
        
        if use_cache and cache_key:
            self._save_to_cache(cache_key, embedding)
        
        return embedding
    
    def encode_batch_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Procesa lote de textos de manera eficiente"""
        self._load_model()
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.model.encode(batch)
            # Normalizar embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
