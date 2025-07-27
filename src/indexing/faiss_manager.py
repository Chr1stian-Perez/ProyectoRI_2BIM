"""
Gestor de índices FAISS para búsqueda vectorial eficiente
"""
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import sys

# Configurar path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# Importaciones absolutas
try:
  from src.utils.config import EMBEDDING_DIMENSION, MODELS_DIR
  from src.utils.logger import setup_logger
except ImportError:
  from utils.config import EMBEDDING_DIMENSION, MODELS_DIR
  from utils.logger import setup_logger

logger = setup_logger(__name__)

class FAISSManager:
  """Gestor de índices FAISS para búsqueda vectorial"""
  
  def __init__(self):
      self.image_index = None
      self.text_index = None
      self.image_metadata = []
      self.text_metadata = []
      self.models_dir = MODELS_DIR
      self.models_dir.mkdir(exist_ok=True)
  
  def create_image_index(self, embeddings: np.ndarray, metadata: List[Dict]) -> faiss.Index:
      """Crea índice FAISS para embeddings de imágenes"""
      logger.info(f"Creando índice de imágenes con {len(embeddings)} embeddings")
      
      # Crear índice FAISS IndexFlatIP para similitud coseno
      index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
      
      # Asegurar que los embeddings estén normalizados
      embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
      
      # Añadir embeddings al índice
      index.add(embeddings.astype(np.float32))
      
      self.image_index = index
      self.image_metadata = metadata
      
      logger.info(f"Índice de imágenes creado: {index.ntotal} vectores")
      return index
  
  def create_text_index(self, embeddings: np.ndarray, metadata: List[Dict]) -> faiss.Index:
      """Crea índice FAISS para embeddings de texto"""
      logger.info(f"Creando índice de texto con {len(embeddings)} embeddings")
      
      # Crear índice FAISS IndexFlatIP para similitud coseno
      index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
      
      # Asegurar que los embeddings estén normalizados
      embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
      
      # Añadir embeddings al índice
      index.add(embeddings.astype(np.float32))
      
      self.text_index = index
      self.text_metadata = metadata
      
      logger.info(f"Índice de texto creado: {index.ntotal} vectores")
      return index
  
  def search_images(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[float], List[Dict]]:
      """Busca imágenes similares usando embedding de consulta"""
      if self.image_index is None:
          raise ValueError("Índice de imágenes no inicializado")
      
      # Normalizar query embedding
      query_embedding = query_embedding / np.linalg.norm(query_embedding)
      query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
      
      # Realizar búsqueda
      similarities, indices = self.image_index.search(query_embedding, k)
      
      # Obtener metadatos correspondientes
      results_metadata = []
      for idx in indices[0]:
          if idx < len(self.image_metadata):
              results_metadata.append(self.image_metadata[idx])
      
      return similarities[0].tolist(), results_metadata
  
  def search_texts(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[float], List[Dict]]:
      """Busca textos similares usando embedding de consulta"""
      if self.text_index is None:
          raise ValueError("Índice de texto no inicializado")
      
      # Normalizar query embedding
      query_embedding = query_embedding / np.linalg.norm(query_embedding)
      query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
      
      # Realizar búsqueda
      similarities, indices = self.text_index.search(query_embedding, k)
      
      # Obtener metadatos correspondientes
      results_metadata = []
      for idx in indices[0]:
          if idx < len(self.text_metadata):
              results_metadata.append(self.text_metadata[idx])
      
      return similarities[0].tolist(), results_metadata
  
  def save_indices(self, image_index_path: str = "image_index.faiss", 
                  text_index_path: str = "text_index.faiss"):
      """Guarda índices FAISS en disco"""
      if self.image_index is not None:
          image_path = self.models_dir / image_index_path
          faiss.write_index(self.image_index, str(image_path))
          
          # Guardar metadatos
          metadata_path = self.models_dir / "image_metadata.pkl"
          with open(metadata_path, 'wb') as f:
              pickle.dump(self.image_metadata, f)
          
          logger.info(f"Índice de imágenes guardado en {image_path}")
      
      if self.text_index is not None:
          text_path = self.models_dir / text_index_path
          faiss.write_index(self.text_index, str(text_path))
          
          # Guardar metadatos
          metadata_path = self.models_dir / "text_metadata.pkl"
          with open(metadata_path, 'wb') as f:
              pickle.dump(self.text_metadata, f)
          
          logger.info(f"Índice de texto guardado en {text_path}")
  
  def load_indices(self, image_index_path: str = "image_index.faiss",
                  text_index_path: str = "text_index.faiss"):
      """Carga índices FAISS desde disco"""
      loaded_image_index = False
      loaded_text_index = False

      try:
          # Cargar índice de imágenes
          image_path = self.models_dir / image_index_path
          if image_path.exists():
              self.image_index = faiss.read_index(str(image_path))
              metadata_path = self.models_dir / "image_metadata.pkl"
              if metadata_path.exists():
                  with open(metadata_path, 'rb') as f:
                      self.image_metadata = pickle.load(f)
                  loaded_image_index = True
                  logger.info(f"Índice de imágenes cargado desde {image_path}")
              else:
                  logger.warning(f"Metadatos de imagen no encontrados")
                  self.image_index = None
          
          # Cargar índice de texto
          text_path = self.models_dir / text_index_path
          if text_path.exists():
              self.text_index = faiss.read_index(str(text_path))
              metadata_path = self.models_dir / "text_metadata.pkl"
              if metadata_path.exists():
                  with open(metadata_path, 'rb') as f:
                      self.text_metadata = pickle.load(f)
                  loaded_text_index = True
                  logger.info(f"Índice de texto cargado desde {text_path}")
              else:
                  logger.warning(f"Metadatos de texto no encontrados")
                  self.text_index = None
              
      except Exception as e:
          logger.error(f"Error cargando índices: {e}")
          return False

      return loaded_image_index and loaded_text_index
