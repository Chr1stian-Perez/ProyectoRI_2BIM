"""
Sistema de recuperación multimodal que integra búsqueda por imagen y texto
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image
import sys
from pathlib import Path

# Configurar path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# Importaciones absolutas
try:
    from src.embeddings.clip_embedder import CLIPEmbedder
    from src.indexing.faiss_manager import FAISSManager
    from src.data_processing.corpus_loader import CorpusLoader
    from src.utils.config import TOP_K_RESULTS, MIN_SIMILARITY_THRESHOLD
    from src.utils.logger import setup_logger
except ImportError:
    from embeddings.clip_embedder import CLIPEmbedder
    from indexing.faiss_manager import FAISSManager
    from data_processing.corpus_loader import CorpusLoader
    from utils.config import TOP_K_RESULTS, MIN_SIMILARITY_THRESHOLD
    from utils.logger import setup_logger

logger = setup_logger(__name__)

class MultimodalRetriever:
    """Sistema de recuperación multimodal para búsqueda por imagen y texto"""
    
    def __init__(self):
        self.embedder = CLIPEmbedder()
        self.faiss_manager = FAISSManager()
        self.corpus_loader = CorpusLoader()
        self.is_initialized = False
    
    def initialize(self):
        """Inicializa el sistema cargando corpus y creando índices"""
        if self.is_initialized:
            return
        
        logger.info("Inicializando sistema de recuperación multimodal")
        # Cargar corpus
        flickr_data = self.corpus_loader.load_flickr8k_real()
        dictionary_data = self.corpus_loader.load_english_dictionary_real()
        
        # Intentar cargar índices existentes
        if self.faiss_manager.load_indices():
            logger.info("Índices FAISS cargados exitosamente")
        else:
            logger.info("Construyendo nuevos índices")
            self._create_indices(flickr_data, dictionary_data)
        
        self.is_initialized = True
        logger.info("Sistema inicializado correctamente")
    
    def _create_indices(self, flickr_data: Dict, dictionary_data: Dict):
        """Crea índices FAISS para imágenes y texto"""
        
        # Preparar datos de imágenes
        image_texts = []
        image_metadata = []
        
        for img_data in flickr_data["images"]:
            main_caption = img_data["captions"][0]
            image_texts.append(main_caption)
            image_metadata.append({
                "filename": img_data["filename"],
                "caption": main_caption,
                "all_captions": img_data["captions"],
                "url": img_data["url"],
                "type": "image"
            })
        
        # Generar embeddings para descripciones de imágenes
        if image_texts:
            image_embeddings = self.embedder.encode_batch_texts(image_texts)
            logger.info(f"Generados {len(image_embeddings)} embeddings de imágenes")
        else:
            image_embeddings = np.array([])
        
        # Crear índice de imágenes
        if image_embeddings.size > 0:
            self.faiss_manager.create_image_index(image_embeddings, image_metadata)
        
        # Preparar datos del diccionario
        dict_texts = []
        dict_metadata = []
        
        for word, data in dictionary_data.items():
            dict_text = f"{word}: {data['definition']}"
            dict_texts.append(dict_text)
            dict_metadata.append({
                "concept": word,
                "definition": data["definition"],
                "characteristics": data["characteristics"],
                "category": data["category"],
                "type": "concept"
            })
        
        # Generar embeddings para diccionario
        if dict_texts:
            dict_embeddings = self.embedder.encode_batch_texts(dict_texts)
            logger.info(f"Generados {len(dict_embeddings)} embeddings del diccionario")
        else:
            dict_embeddings = np.array([])
        
        # Crear índice de texto (combinando imágenes y diccionario)
        all_text_embeddings = np.array([])
        all_text_metadata = []

        if image_embeddings.size > 0 and dict_embeddings.size > 0:
            all_text_embeddings = np.vstack([image_embeddings, dict_embeddings])
            all_text_metadata = image_metadata + dict_metadata
        elif image_embeddings.size > 0:
            all_text_embeddings = image_embeddings
            all_text_metadata = image_metadata
        elif dict_embeddings.size > 0:
            all_text_embeddings = dict_embeddings
            all_text_metadata = dict_metadata

        if all_text_embeddings.size > 0:
            self.faiss_manager.create_text_index(all_text_embeddings, all_text_metadata)
        
        # Guardar índices
        self.faiss_manager.save_indices()
    
    def search_by_image(self, image: Union[Image.Image, str], k: int = TOP_K_RESULTS) -> Dict:
        """Realiza búsqueda usando una imagen como consulta"""
        if not self.is_initialized:
            self.initialize()
        
        logger.info("--Búsqueda por imagen--")
        
        # Generar embedding de la imagen
        image_embedding = self.embedder.encode_image(image)
        logger.info(f"Embedding de imagen generado: shape {image_embedding.shape}")
        
        # Buscar imágenes similares
        img_similarities, img_results = self.faiss_manager.search_images(image_embedding, k)
        logger.info(f"Búsqueda de imágenes: {len(img_similarities)} similitudes, {len(img_results)} resultados")
        
        # Buscar conceptos relacionados
        concept_similarities, concept_results_raw = self.faiss_manager.search_texts(image_embedding, k)
        logger.info(f"Búsqueda de conceptos: {len(concept_similarities)} similitudes, {len(concept_results_raw)} resultados")
        
        # Filtrar resultados por umbral de similitud
        filtered_img_results = []
        for sim, result in zip(img_similarities, img_results):
            logger.debug(f"Imagen: similitud={sim}, umbral={MIN_SIMILARITY_THRESHOLD}")
            if sim >= MIN_SIMILARITY_THRESHOLD:
                result["similarity"] = float(sim)
                filtered_img_results.append(result)
        
        filtered_concept_results = []
        for sim, result in zip(concept_similarities, concept_results_raw):
            logger.debug(f"Concepto: similitud={sim}, tipo={result.get('type')}, umbral={MIN_SIMILARITY_THRESHOLD}")
            if sim >= MIN_SIMILARITY_THRESHOLD and result.get("type") == "concept":
                result["similarity"] = float(sim)
                filtered_concept_results.append(result)

        logger.info(f"Resultados filtrados: {len(filtered_img_results)} imágenes, {len(filtered_concept_results)} conceptos")

        return {
            "query_type": "image",
            "similar_images": filtered_img_results,
            "related_concepts": filtered_concept_results[:3],
            "total_results": len(filtered_img_results) + len(filtered_concept_results)
        }
    
    def search_by_text(self, query: str, k: int = TOP_K_RESULTS) -> Dict:
        """Realiza búsqueda usando texto como consulta"""
        if not self.is_initialized:
            self.initialize()
        
        logger.info(f"Realizando búsqueda por texto: '{query}'")
        
        # Generar embedding del texto
        text_embedding = self.embedder.encode_text(query)
        
        # Buscar en índice de texto
        similarities, results = self.faiss_manager.search_texts(text_embedding, k * 2)
        
        # Separar resultados por tipo y aplicar umbral de similitud
        image_results = []
        concept_results = []
        
        for sim, result in zip(similarities, results):
            if sim >= MIN_SIMILARITY_THRESHOLD:
                result["similarity"] = float(sim)
                if result.get("type") == "image":
                    image_results.append(result)
                elif result.get("type") == "concept":
                    concept_results.append(result)
        
        # Limitar resultados
        image_results = image_results[:k]
        concept_results = concept_results[:3]

        return {
            "query_type": "text",
            "query": query,
            "related_images": image_results,
            "related_concepts": concept_results,
            "total_results": len(image_results) + len(concept_results)
        }
    
    def get_context_for_generation(self, search_results: Dict) -> str:
        """Prepara contexto para generación de respuesta con Gemini"""
        logger.info(f"Generando contexto desde resultados: {search_results.keys()}")
        
        context_parts = []
        
        # Agregar información de imágenes
        if "similar_images" in search_results and search_results["similar_images"]:
            logger.info(f"Agregando {len(search_results['similar_images'])} imágenes similares al contexto")
            context_parts.append("=== IMÁGENES SIMILARES ===")
            for i, img in enumerate(search_results["similar_images"], 1):
                context_parts.append(f"{i}. {img['caption']} (similitud: {img['similarity']:.3f})")
        
        if "related_images" in search_results and search_results["related_images"]:
            logger.info(f"Agregando {len(search_results['related_images'])} imágenes relacionadas al contexto")
            context_parts.append("=== IMÁGENES RELACIONADAS ===")
            for i, img in enumerate(search_results["related_images"], 1):
                context_parts.append(f"{i}. {img['caption']} (similitud: {img['similarity']:.3f})")
        
        # Agregar información de conceptos
        if "related_concepts" in search_results and search_results["related_concepts"]:
            logger.info(f"Agregando {len(search_results['related_concepts'])} conceptos relacionados al contexto")
            context_parts.append("\n=== CONCEPTOS RELACIONADOS ===")
            for i, concept in enumerate(search_results["related_concepts"], 1):
                context_parts.append(f"{i}. {concept['concept']}: {concept['definition']}")
        
        context = "\n".join(context_parts)
        logger.info(f"Contexto generado (longitud: {len(context)}): {context[:200]}...")
        
        # Validar que el contexto no esté vacío
        if not context or context.strip() == "":
            logger.error("Contexto vacío generado")
            logger.error(f"Resultados de búsqueda: {search_results}")
            raise ValueError("No se encontraron resultados relevantes para generar contexto")
        
        return context
