"""
Cargador y procesador de los corpus Flickr8k y English Dictionary
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import hashlib
import shutil
from PIL import Image
import requests
from io import BytesIO
import zipfile
import os
import kagglehub
from tqdm import tqdm
import csv
import sys

# Configurar path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# Importaciones absolutas
try:
    from src.utils.config import DATA_DIR, CACHE_DIR, FLICKR8K_MAX_IMAGES, DICTIONARY_MAX_ENTRIES
    from src.utils.logger import setup_logger
except ImportError:
    from utils.config import DATA_DIR, CACHE_DIR, FLICKR8K_MAX_IMAGES, DICTIONARY_MAX_ENTRIES
    from utils.logger import setup_logger

logger = setup_logger(__name__)

class CorpusLoader:
    """Gestor de carga y procesamiento de corpus multimodal reales"""
    
    def __init__(self):
        self.flickr8k_data = None
        self.dictionary_data = None
        self.flickr8k_path = DATA_DIR / "flickr8k"
        self.dictionary_path = DATA_DIR / "english-dictionary"
        self.image_cache = {}
        
        # Crear directorios
        self.flickr8k_path.mkdir(exist_ok=True)
        self.dictionary_path.mkdir(exist_ok=True)
    
    def download_flickr8k(self) -> bool:
        """Descarga el dataset Flickr8k desde Kaggle usando kagglehub"""
        logger.info("Descargando dataset Flickr8k desde Kaggle usando kagglehub")
        
        try:
            # Verificar si ya existe
            if (self.flickr8k_path / "captions.txt").exists() or \
               (self.flickr8k_path / "Flickr8k.token.txt").exists():
                logger.info("Dataset Flickr8k ya existe localmente")
                return True
        
            # Usar kagglehub para descargar
            # Descargar dataset usando kagglehub
            download_path = kagglehub.dataset_download("adityajn105/flickr8k")
            logger.info(f"Dataset descargado en: {download_path}")
        
            # Copiar archivos al directorio del proyecto     
            download_path = Path(download_path)
        
            # Copiar todos los archivos al directorio del proyecto
            for item in download_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, self.flickr8k_path / item.name)
                elif item.is_dir():
                    shutil.copytree(item, self.flickr8k_path / item.name, dirs_exist_ok=True)
        
            logger.info("Dataset Flickr8k descargado exitosamente")
            return True
    
        except Exception as e:
            logger.error(f"Error descargando Flickr8k: {e}")
            logger.error("Configure las credenciales de Kaggle correctamente")
            return False

    
    def download_english_dictionary(self) -> bool:
        """Descarga el diccionario inglés desde Kaggle usando kagglehub"""
        logger.info("Descargando English Dictionary desde Kaggle usando kagglehub...")
        
        try:
            # Verificar si ya existe
            csv_files = list(self.dictionary_path.glob("*.csv"))
            if csv_files:
                logger.info("English Dictionary ya existe localmente")
                return True
            
            # Descargar dataset usando kagglehub
            download_path = kagglehub.dataset_download("anthonytherrien/larger-dictionary-of-english-words-and-definitions")
            logger.info(f"Dataset descargado en: {download_path}")
            
            # Copiar archivos al directorio del proyecto
            import shutil
            from pathlib import Path
            
            download_path = Path(download_path)
            
            # Buscar archivos CSV en el directorio descargado
            csv_files = list(download_path.glob("*.csv"))
            if not csv_files:
                logger.error("No se encontraron archivos CSV en el dataset descargado")
                return False
            
            # Copiar el primer archivo CSV encontrado
            source_csv = csv_files[0]
            target_csv = self.dictionary_path / "dictionary.csv"
            
            shutil.copy2(source_csv, target_csv)
            logger.info(f"Archivo copiado a: {target_csv}")
            
            logger.info("English Dictionary descargado exitosamente")
            return True
        
        except Exception as e:
            logger.error(f"Error descargando English Dictionary: {e}")
            logger.error("Configure las credenciales de Kaggle correctamente")
            return False
    
    def load_flickr8k_real(self) -> Dict:
        """Carga el dataset real de Flickr8k"""
        logger.info("Cargando dataset real Flickr8k")
        
        if not self.download_flickr8k():
            logger.error("No se pudo descargar Flickr8k - verifique credenciales de Kaggle")
            raise RuntimeError("Dataset Flickr8k no disponible")
        
        try:
            # Cargar captions
            captions_file = self.flickr8k_path / "captions.txt"
            if not captions_file.exists():
                # Buscar archivo alternativo
                possible_files = ["Flickr8k.token.txt", "results.csv"]
                found_caption_file = None
                for filename in possible_files:
                    if (self.flickr8k_path / filename).exists():
                        found_caption_file = self.flickr8k_path / filename
                        break
                
                if found_caption_file is None:
                    logger.error("No se encontró archivo de captions para Flickr8k")
                    return {"images": []}
                captions_file = found_caption_file
            
            # Leer captions
            captions_data = {}
            
            if captions_file.name == "captions.txt":
                df = pd.read_csv(captions_file)
                for _, row in df.iterrows():
                    image_name = row['image']
                    caption = row['caption']
                    
                    if image_name not in captions_data:
                        captions_data[image_name] = []
                    captions_data[image_name].append(caption)
            
            else:
                # Formato alternativo
                with open(captions_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if '\t' in line:
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                image_caption = parts[0]
                                caption = parts[1]
                                
                                if '#' in image_caption:
                                    image_name = image_caption.split('#')[0]
                                else:
                                    image_name = image_caption
                                
                                if image_name not in captions_data:
                                    captions_data[image_name] = []
                                captions_data[image_name].append(caption)
            
            # Convertir a formato del sistema
            images_data = []
            images_dir = self.flickr8k_path / "Images"
            
            if not images_dir.exists():
                possible_dirs = ["images", "Flicker8k_Dataset"]
                for dirname in possible_dirs:
                    if (self.flickr8k_path / dirname).exists():
                        images_dir = self.flickr8k_path / dirname
                        break
            
            # Limitar a FLICKR8K_MAX_IMAGES para rendimiento
            for image_name, captions in list(captions_data.items())[:FLICKR8K_MAX_IMAGES]:
                image_path = images_dir / image_name if images_dir.exists() else None
                
                if image_path and image_path.exists():
                    image_url = str(image_path)
                else:
                    image_url = f"/placeholder.svg?height=300&width=400"
                
                images_data.append({
                    "filename": image_name,
                    "captions": captions[:5],
                    "url": image_url,
                    "local_path": str(image_path) if image_path and image_path.exists() else None
                })
            
            flickr_data = {"images": images_data}
            self.flickr8k_data = flickr_data
            
            logger.info(f"Dataset Flickr8k cargado: {len(images_data)} imágenes")
            return flickr_data
            
        except Exception as e:
            logger.error(f"Error cargando Flickr8k real: {e}")
            return {"images": []}
    
    def load_english_dictionary_real(self) -> Dict:
        """Carga el diccionario inglés real desde Kaggle"""
        logger.info("Cargando corpus: English Dictionary")
        
        if not self.download_english_dictionary():
            logger.error("No se pudo descargar English Dictionary - verifique credenciales de Kaggle")
            raise RuntimeError("English Dictionary no disponible")
        
        try:
            dictionary_data = {}
            
            # Buscar archivo CSV
            csv_files = list(self.dictionary_path.glob("*.csv"))
            if not csv_files:
                logger.error("No se encontró archivo CSV del diccionario")
                return {}
            
            csv_file = csv_files[0]  # Usar el primer archivo CSV encontrado
            logger.info(f"Cargando diccionario desde: {csv_file}")
            
            # Cargar datos del CSV
            df = pd.read_csv(csv_file)
            logger.info(f"CSV cargado con {len(df)} filas y columnas: {df.columns.tolist()}")
            
            # Verificar columnas esperadas
            word_col = None
            def_col = None
            
            # Buscar columnas de palabra y definición
            for col in df.columns:
                col_lower = col.lower()
                if 'word' in col_lower and not word_col:
                    word_col = col
                elif any(term in col_lower for term in ['definition', 'meaning', 'def']) and not def_col:
                    def_col = col
            
            if not word_col or not def_col:
                logger.error(f"Columnas no encontradas. Columnas disponibles: {df.columns.tolist()}")
                logger.error(f"Buscando: word_col={word_col}, def_col={def_col}")
                return {}
            
            logger.info(f"Usando columnas: palabra='{word_col}', definición='{def_col}'")
            
            # Procesar datos limitando a DICTIONARY_MAX_ENTRIES
            processed_count = 0
            skipped_count = 0
            
            for _, row in df.iterrows():
                if processed_count >= DICTIONARY_MAX_ENTRIES:
                    break
                
                try:
                    word = str(row[word_col]).lower().strip()
                    definition = str(row[def_col]).strip()
                    
                    # Validar datos
                    if not word or word == 'nan' or not definition or definition == 'nan':
                        skipped_count += 1
                        continue
                    
                    if len(definition) < 10:  # Definiciones muy cortas
                        skipped_count += 1
                        continue
                    
                    dictionary_data[word] = {
                        "definition": definition,
                        "characteristics": self._extract_characteristics(definition),
                        "category": "english_word",
                        "source": "kaggle_dictionary"
                    }
                    processed_count += 1
                    
                except Exception as e:
                    skipped_count += 1
                    continue
            
            self.dictionary_data = dictionary_data
            logger.info(f"English Dictionary cargado: {len(dictionary_data)} palabras procesadas, {skipped_count} omitidas")
            
            # Prueba: Verificar si 'dog' está en el diccionario
            if 'dog' in dictionary_data:
                logger.info(f"✅ Palabra 'dog' encontrada: {dictionary_data['dog']['definition'][:100]}...")
            else:
                logger.warning("Palabra 'dog' no encontrada en el diccionario")
            
            return dictionary_data
            
        except Exception as e:
            logger.error(f"Error cargando English Dictionary: {e}")
            return {}
    
    def _extract_characteristics(self, definition: str) -> List[str]:
        """Extrae características clave de una definición"""
        keywords = ['is', 'are', 'has', 'have', 'can', 'used', 'type', 'kind', 'form']
        characteristics = []
        
        words = definition.lower().split()
        for i, word in enumerate(words):
            if word in keywords and i < len(words) - 1:
                char = ' '.join(words[i+1:i+4])
                characteristics.append(char)
        
        if not characteristics:
            characteristics = words[:5]
        
        return characteristics[:5]
    
    # Métodos de compatibilidad
    def load_3d_ex_real(self) -> Dict:
        """Método de compatibilidad - ahora carga el diccionario inglés"""
        return self.load_english_dictionary_real()
    
    def load_flickr8k_sample(self) -> Dict:
        """Alias para compatibilidad"""
        return self.load_flickr8k_real()
    
    def get_image_descriptions(self) -> List[str]:
        """Retorna todas las descripciones de imágenes del corpus"""
        if not self.flickr8k_data:
            self.load_flickr8k_real()
        
        descriptions = []
        if self.flickr8k_data and "images" in self.flickr8k_data:
            for image_data in self.flickr8k_data["images"]:
                descriptions.extend(image_data["captions"])
        
        return descriptions
    
    def get_concept_definitions(self) -> List[str]:
        """Retorna todas las definiciones del diccionario"""
        if not self.dictionary_data:
            self.load_english_dictionary_real()
        
        definitions = []
        if self.dictionary_data:
            for word, data in self.dictionary_data.items():
                definitions.append(f"{word}: {data['definition']}")
        
        return definitions
    
    def find_concept_by_keywords(self, keywords: List[str]) -> Optional[Dict]:
        """Encuentra palabras relacionadas con palabras clave"""
        if not self.dictionary_data:
            self.load_english_dictionary_real()
        
        if self.dictionary_data:
            for word, data in self.dictionary_data.items():
                if any(keyword.lower() in word.lower() or 
                       any(keyword.lower() in char.lower() for char in data["characteristics"])
                       for keyword in keywords):
                    return {word: data}
        
        return None