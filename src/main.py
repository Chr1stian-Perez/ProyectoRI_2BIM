"""
Sistema RAG Multimodal - Aplicación principal
"""
import streamlit as st
import sys
import os
from pathlib import Path

# Configurar paths para importaciones
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

# Importaciones absolutas
try:
    from src.retrieval.multimodal_retriever import MultimodalRetriever
    from src.generation.gemini_generator import GeminiGenerator
    from src.utils.logger import setup_logger
    from src.utils.config import SUPPORTED_IMAGE_FORMATS, MAX_FILE_SIZE_MB
except ImportError:
    # Importaciones alternativas
    from retrieval.multimodal_retriever import MultimodalRetriever
    from generation.gemini_generator import GeminiGenerator
    from utils.logger import setup_logger
    from utils.config import SUPPORTED_IMAGE_FORMATS, MAX_FILE_SIZE_MB

logger = setup_logger(__name__)

class MultimodalRAGApp:
    """Aplicación principal del sistema RAG multimodal"""
    
    def __init__(self):
        # Inicializar componentes en session_state para persistencia
        if 'retriever' not in st.session_state:
            st.session_state.retriever = None
        if 'generator' not in st.session_state:
            st.session_state.generator = None
        if 'show_results' not in st.session_state:
            st.session_state.show_results = False
        if 'last_search_results' not in st.session_state:
            st.session_state.last_search_results = None
        if 'last_generated_response' not in st.session_state:
            st.session_state.last_generated_response = None

        self._initialize_components()
  
    def _initialize_components(self):
        """Inicializa componentes del sistema si no están ya en session_state"""
        if st.session_state.retriever is None:
            with st.spinner("Inicio del sistema RAG multimodal"):
                try:
                    st.session_state.retriever = MultimodalRetriever()
                    st.session_state.retriever.initialize()
                    st.session_state.generator = GeminiGenerator()
                except Exception as e:
                    st.error(f"Error inicializando sistema: {e}")
                    st.error("Verificar:")
                    st.error("1. GEMINI_API_KEY configurada en .env")
                    st.error("2. Los datasets estén descargados correctamente en /data")
                    st.stop()
        
        self.retriever = st.session_state.retriever
        self.generator = st.session_state.generator
  
    def render_header(self):
        """Renderiza el encabezado de la aplicación"""
        st.set_page_config(
            page_title="ProyectoRI_2BIM - Sistema RAG Multimodal",
            page_icon="📚​",
            layout="wide"
        )
        
        st.title("​📚​Proyecto Retrieval Information 2do Bimestre​")
        st.subheader("Sistema de Recuperación de Información Multimodal 🗿")
        
        st.markdown("""
        **Autores:** Fricxon Pambabay, Christian Pérez, Jeremmy Perugachi  
        **Asignatura:** ICCD753 - Recuperación de Información  
        **Institución:** Escuela Politécnica Nacional
        """)
        
        st.divider()
  
    def render_search_interface(self):
        """Renderiza la interfaz de búsqueda"""
        st.markdown("### 🌐 Búsqueda Multimodal")
        
        # Selector de modo de búsqueda
        search_mode = st.radio(
            "Seleccione el modo de búsqueda:",
            ["🖼️ Búsqueda por Imagen", "📝 Búsqueda por Texto"],
            horizontal=True,
            key="search_mode_radio" # Añadir una clave única
        )
        # Resetear resultados si el modo de búsqueda cambia
        if st.session_state.get('prev_search_mode', search_mode) != search_mode:
            st.session_state.show_results = False
            st.session_state.last_search_results = None
            st.session_state.last_generated_response = None
        st.session_state.prev_search_mode = search_mode

        if search_mode == "🖼️ Búsqueda por Imagen":
            self._render_image_search()
        else:
            self._render_text_search()
        
        # Mostrar resultados si están disponibles
        if st.session_state.show_results and st.session_state.last_search_results:
            self._display_results(st.session_state.last_search_results, st.session_state.last_generated_response)
  
    def _render_image_search(self):
        """Renderiza interfaz de búsqueda por imagen"""
        st.markdown("#### Cargar Imagen")
        
        uploaded_file = st.file_uploader(
            "Seleccione una imagen para analizar",
            type=SUPPORTED_IMAGE_FORMATS,
            help=f"Formatos soportados: {', '.join(SUPPORTED_IMAGE_FORMATS)}. Tamaño máximo: {MAX_FILE_SIZE_MB}MB",
            key="image_uploader" # Añadir una clave única
        )
        
        if uploaded_file is not None:
            # Validar tamaño de archivo
            if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                st.error(f"El archivo es demasiado grande. Máximo permitido: {MAX_FILE_SIZE_MB}MB")
                return
            
            # Mostrar imagen cargada
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(uploaded_file, caption="Imagen cargada", use_container_width=True)
            
            with col2:
                # Usar un formulario para el botón de análisis de imagen
                with st.form("image_search_form", clear_on_submit=False): # No limpiar para que la imagen persista
                    submitted_image = st.form_submit_button("​👀​ Analizar Imagen", type="primary", use_container_width=True)
                    if submitted_image:
                        self._process_image_search(uploaded_file)
  
    def _render_text_search(self):
        """Renderiza interfaz de búsqueda por texto con botón siempre visible"""
        st.markdown("#### Consulta Textual")
        
        with st.form("text_search_form", clear_on_submit=True):
            col1, col2 = st.columns([0.9, 0.1]) # Ajustar proporciones para el input y el botón
            
            with col1:
                query = st.text_input(
                    "Ingrese su consulta:",
                    placeholder="Ej: perro corriendo en el parque, montañas nevadas, niños jugando...",
                    label_visibility="collapsed", # Ocultar la etiqueta para un diseño más compacto
                    key="text_query_input" # Añadir una clave única
                )
            
            with col2:
                # El botón de submit del formulario siempre es visible
                submitted_text = st.form_submit_button("🔍", type="primary", use_container_width=True)
            
            if submitted_text:
                if query:
                    self._process_text_search(query)
                else:
                    st.warning("Por favor, ingrese una consulta.")
    
    def _process_image_search(self, uploaded_file):
        """Procesa búsqueda por imagen"""
        try:
            with st.spinner("--Analizando imagen--"):
                from PIL import Image
                image = Image.open(uploaded_file)
                search_results = self.retriever.search_by_image(image)
                
                try:
                    context = self.retriever.get_context_for_generation(search_results)
                    logger.debug(f"Contexto enviado a Gemini (búsqueda por imagen): {context}")
                    response = self.generator.generate_response("", context, "image")
                except ValueError as e:
                    st.error(f"No se encontraron resultados relevantes: {e}")
                    return
                except RuntimeError as e:
                    st.error(f"Error generando respuesta: {e}")
                    return
                
                # Almacenar resultados en session_state y activar la visualización
                st.session_state.last_search_results = search_results
                st.session_state.last_generated_response = response
                st.session_state.show_results = True
                st.rerun() # Forzar un re-run para mostrar los resultados
                
        except Exception as e:
            st.error(f"Error procesando imagen: {str(e)}")
            logger.error(f"Error en búsqueda por imagen: {e}")
  
    def _process_text_search(self, query):
        """Procesa búsqueda por texto"""
        try:
            with st.spinner("Buscando información relevante..."):
                search_results = self.retriever.search_by_text(query)
                
                try:
                    context = self.retriever.get_context_for_generation(search_results)
                    logger.debug(f"Contexto enviado a Gemini (búsqueda por texto): {context}")
                    response = self.generator.generate_response(query, context, "text")
                except ValueError as e:
                    st.error(f"No se encontraron resultados relevantes: {e}")
                    return
                except RuntimeError as e:
                    st.error(f"Error generando respuesta: {e}")
                    return
                
                # Almacenar resultados en session_state y activar la visualización
                st.session_state.last_search_results = search_results
                st.session_state.last_generated_response = response
                st.session_state.show_results = True
                st.rerun() # Forzar un re-run para mostrar los resultados
                
        except Exception as e:
            st.error(f"Error procesando consulta: {str(e)}")
            logger.error(f"Error en búsqueda por texto: {e}")
  
    def _display_results(self, search_results, response):
        """Muestra los resultados de búsqueda"""
        st.markdown("### -- Resultados --")
        
        # Mostrar respuesta generada
        st.markdown("#### Respuesta Generada por el RAG")
        st.markdown(response)
        
        # Mostrar métricas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Resultados", search_results.get("total_results", 0))
        with col2:
            images_count = len(search_results.get("similar_images", [])) + len(search_results.get("related_images", []))
            st.metric("Imágenes Encontradas", images_count)
        with col3:
            concepts_count = len(search_results.get("related_concepts", []))
            st.metric("Conceptos Relacionados", concepts_count)
        
        # Mostrar imágenes similares/relacionadas
        images_key = "similar_images" if "similar_images" in search_results else "related_images"
        if images_key in search_results and search_results[images_key]:
            st.markdown("#### 🖼️ Imágenes Relevantes")
            
            for i, img_data in enumerate(search_results[images_key][:5]):
                with st.expander(f"Imagen {i+1} - Similitud: {img_data['similarity']:.3f}"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.image(img_data["url"], use_container_width=True)
                    
                    with col2:
                        st.write(f"**Descripción:** {img_data['caption']}")
                        if 'all_captions' in img_data:
                            st.write("**Descripciones adicionales:**")
                            for caption in img_data['all_captions'][1:]:
                                st.write(f"• {caption}")
      
        # Mostrar conceptos relacionados
        if "related_concepts" in search_results and search_results["related_concepts"]:
            st.markdown("#### 📚 Conceptos Relacionados")
            
            for concept_data in search_results["related_concepts"]:
                with st.expander(f"{concept_data['concept'].title()} - Similitud: {concept_data['similarity']:.3f}"):
                    st.write(f"**Definición:** {concept_data['definition']}")
                    st.write(f"**Categoría:** {concept_data['category']}")
                    st.write(f"**Características:** {', '.join(concept_data['characteristics'])}")
      
        # Sistema de feedback (ahora con botones normales, no en un formulario)
        self._render_feedback_section()
  
    def _render_feedback_section(self):
        """Renderiza sección de feedback con botones normales"""
        st.markdown("#### 💬 Evaluación de Resultados")
        
        col1, col2 = st.columns(2)
        with col1:
            # Usar st.button normal, ya que no está anidado en un formulario
            if st.button("👍 Útil", use_container_width=True, key="feedback_useful_btn"):
                st.success("¡Gracias por su feedback positivo!")
        with col2:
            # Usar st.button normal
            if st.button("👎 No útil", use_container_width=True, key="feedback_not_useful_btn"):
                st.info("Gracias por su feedback. Trabajamos para mejorar.")
  
    def render_sidebar(self):
        """Renderiza barra lateral con información del sistema"""
        with st.sidebar:
            st.markdown("### ℹ️ Información del Sistema")
            
            st.markdown("""
            **Arquitectura:**
            - ​🪄​ Modelo: CLIP ViT-B/32
            - 📋 Índice: FAISS IndexFlatIP
            - 🤖Generación: Gemini 2.0 Flash
            - ​📋​ Dimensiones: 512D
            
            **Corpus:**
            - 📸 Flickr8k (imágenes + descripciones)
            - 📚 English Dictionary (conceptos + definiciones)
            
            **Características:**
            - ✅ Búsqueda multimodal
            - ✅ Similitud semántica
            - ✅ Generación contextual
            - ✅ Interfaz interactiva
            """)
            
            st.divider()
            
            st.markdown("### Configuración")
            
            # Mostrar estado del sistema
            if hasattr(st.session_state, 'retriever') and st.session_state.retriever is not None:
                st.success("✅ Sistema inicializado")
            else:
                st.warning("⏳ Inicializando el sistema ⏳")
            
            # Información de rendimiento
            st.markdown("**Métricas:**")
            st.write("• Umbral similitud: 0.05")
            st.write("• Top-K resultados: 5")
            st.write("• Formatos: JPG, PNG, WEBP, BMP")
            st.write("• Tamaño máx: 10MB")
    
    def run(self):
        """Ejecuta la aplicación"""
        self.render_header()
        self.render_sidebar()
        self.render_search_interface()

def main():
    """Función principal"""
    app = MultimodalRAGApp()
    app.run()

if __name__ == "__main__":
    main()
