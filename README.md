# Proyecto Retrieval Information 2BIM - Sistema RAG Multimodal

Sistema de Recuperación de Información multimodal que implementa búsqueda por imagen y texto utilizando arquitectura RAG (Retrieval Augmented Generation) con **corpus Flickr8k y English Dictionary**.

##  Autores

- **Fricxon Pambabay** - fricxon.pambabay@epn.edu.ec
- **Christian Pérez** - christian.perez01@epn.edu.ec  
- **Jeremmy Perugachi** - jeremmy.perugachi@epn.edu.ec

**Institución:** Escuela Politécnica Nacional  
**Facultad:** Ingeniería de Sistemas  
**Asignatura:** ICCD753 - Recuperación de Información  
**Período:** 2025A

##  Arquitectura del sistema

### Componentes principales

1. **Embeddings multimodales**: CLIP ViT-B/32 para generar representaciones vectoriales de 512 dimensiones
2. **Indexación vectorial**: FAISS IndexFlatIP para búsqueda eficiente por similitud coseno
3. **Corpus multimodal**: 
   - Flickr8k (imágenes con descripciones)
   - English Dictionary (conceptos y definiciones)
4. **Generación de Respuestas**: Gemini 2.0 Flash para síntesis contextual
5. **Interfaz de Usuario**: Streamlit 

### Flujo de Procesamiento

```
Consulta (Imagen/Texto) → Embedding CLIP → Búsqueda FAISS → Contexto → Gemini → Respuesta
```

## Instalación y Configuración

### Prerrequisitos

- Python 3.8+
- Cuenta de Kaggle (para descargar Flickr8k)
- pip y Git

### Instalación Paso a Paso

1. **Clonar el repositorio**
\`\`\`bash
git clone <repository-url>
cd ProyectoRI_2BIM
\`\`\`

2. **Ejecutar instalación completa**
\`\`\`bash
python setup.py
\`\`\`

Este script único:
- ✅ Verifica Python y pip
- ✅ Crea estructura de directorios
- ✅ Instala dependencias compatibles
- ✅ Corrige importaciones automáticamente
- ✅ Descarga datasets (Flickr8k + English Dictionary)
- ✅ Configura variables de entorno
- ✅ Prueba la instalación completa

3. **Configurar API keys (opcional)**
\`\`\`bash
# Editar .env con sus keys
nano .env
# Configurar:
GEMINI_API_KEY=your_actual_gemini_key
KAGGLE_USERNAME=your_kaggle_username  
KAGGLE_KEY=your_kaggle_key
\`\`\`

4. **Ejecutar aplicación**
\`\`\`bash
python run.py
\`\`\`

### Instalación Automática

El script `setup.py` maneja automáticamente:
- **Compatibilidad de Python**: Detecta versión e instala PyTorch compatible
- **Resolución de Errores**: Corrige importaciones y dependencias
- **Descarga de Datos**: Obtiene los datasets automáticamente
- **Configuración Completa**: Prepara todo el entorno de trabajo

### Obtener API Keys

#### Gemini API Key
1. Vaya a [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Inicie sesión con su cuenta de Google
3. Haga clic en "Create API Key"
4. Copie la clave generada

#### Kaggle API Credentials
1. Vaya a [Kaggle Account](https://www.kaggle.com/account)
2. Scroll hasta "API" section
3. Haga clic en "Create New API Token"
4. Descargue kaggle.json o use username/key directamente

### Ejecución

\`\`\`bash
streamlit run app.py
\`\`\`

La aplicación estará disponible en `http://localhost:8501`

## 📁 Estructura del Proyecto

\`\`\`
ProyectoRI_2BIM/
├── src/
│   ├── data_processing/     # Carga y procesamiento de corpus
│   ├── embeddings/          # Generación de embeddings CLIP
│   ├── indexing/           # Gestión de índices FAISS
│   ├── retrieval/          # Sistema de recuperación multimodal
│   ├── generation/         # Integración con Gemini
│   └── utils/              # Configuración y utilidades
├── scripts/                # Scripts de configuración
├── data/                   # Datos del corpus
├── models/                 # Modelos e índices guardados
├── cache/                  # Caché de embeddings
├── app.py                  # Punto de entrada principal
├── requirements.txt        # Dependencias Python
└── README.md              # Documentación
\`\`\`

## 📁 Estructura de Datos

\`\`\`
data/
├── flickr8k/
│   ├── captions.txt          # Descripciones de imágenes
│   ├── Images/               # 8,000 imágenes JPG
│   └── Flickr8k.token.txt   # Formato alternativo de captions
└── english_dictionary/
    ├── english_dictionary.csv            # Definiciones en formato CSV
\`\`\`

##  Configuración Técnica

### Especificaciones del Modelo

- **CLIP**: ViT-B/32 (Visual Transformer Base, patches 32x32)
- **Dimensionalidad**: 512 dimensiones
- **Biblioteca**: SentenceTransformer('clip-ViT-B-32')
- **Normalización**: L2 para similitud coseno

### Configuración FAISS

- **Tipo de Índice**: IndexFlatIP (Inner Product)
- **Optimización**: Similitud coseno exacta
- **Escalabilidad**: Hasta 10K vectores eficientemente

### Parámetros de Búsqueda

- **Top-K Resultados**: 5
- **Umbral de Similitud**: 0.1
- **Formatos Soportados**: JPG, PNG, WEBP, BMP
- **Tamaño Máximo**: 10MB

##  Funcionalidades

### Búsqueda por Imagen

1. **Carga de Imagen**: Interfaz drag-and-drop
2. **Procesamiento**: Generación de embedding visual
3. **Recuperación**: Búsqueda de imágenes similares
4. **Identificación**: Mapeo a conceptos textuales
5. **Generación**: Respuesta contextual educativa

### Búsqueda por Texto

1. **Consulta Textual**: Campo de entrada intuitivo
2. **Embedding Textual**: Representación vectorial
3. **Búsqueda Cruzada**: En corpus visual y textual
4. **Ranking**: Por relevancia semántica
5. **Síntesis**: Respuesta multimodal integrada

## Corpus de Datos Reales

### Flickr8k Dataset
- **Fuente**: [Kaggle - Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- **Contenido**: 8,000 imágenes reales con 5 descripciones cada una (40,000 captions)
- **Dominio**: Escenas cotidianas, personas, animales, objetos, paisajes
- **Formato**: Imágenes JPG + archivo de captions CSV/TXT
- **Tamaño**: ~1GB (imágenes + metadatos)

### English Dictionary Dataset
- **Fuente**: [Kaggle - Larger Dictionary of English Words and Definitions](https://www.kaggle.com/datasets/anthonytherrien/larger-dictionary-of-english-words-and-definitions)
- **Contenido**: 42,052 palabras en inglés con sus definiciones correspondientes
- **Dominio**: Vocabulario completo desde términos comunes hasta palabras especializadas
- **Formato**: CSV con columnas word/definition
- **Tamaño**: ~5MB
- **Calidad**: Definiciones detalladas y precisas para cada palabra

##  Configuración de Corpus

### Limitaciones de Rendimiento
- **Flickr8k**: Limitado a 1,500 imágenes para rendimiento inicial
- **English Dictionary**: Filtrado de definiciones con mínimo 10 caracteres
- **Caché**: Embeddings persistentes para evitar recálculos

### Formato de Datos

**Flickr8k**:
\`\`\`
image,caption
1000268201_693b08cb0e.jpg,"A child in a pink dress is climbing up a set of stairs in an entry way ."
1000268201_693b08cb0e.jpg,"A girl going into a wooden building ."
\`\`\`

**English Dictionary**:
\`\`\`json
{
  "word": "dog",
  "definition": "A domesticated carnivorous mammal...",
  "category": "animal"
}
\`\`\`

## 🔍 Metodología de Evaluación

### Métricas Cuantitativas

- **Similitud Coseno**: Medida primaria de relevancia
- **Tiempo de Respuesta**: < 3 segundos objetivo
- **Precisión**: Top-K accuracy en recuperación

### Sistema de Feedback

- **Interfaz de Rating**: Thumbs up/down
- **Logging Automático**: Para análisis posterior
- **Casos Edge**: Documentación sistemática

##  Optimizaciones

### Gestión de Memoria

- **Lazy Loading**: Carga diferida de modelos
- **Caché Inteligente**: Embeddings precomputados
- **Normalización**: Optimización de cálculos

### Rendimiento

- **Batch Processing**: Procesamiento por lotes
- **GPU Acceleration**: Cuando disponible
- **Índices Persistentes**: Almacenamiento en disco

##  Casos de Uso

### Educativo
- Identificación de objetos en imágenes
- Explicaciones contextuales detalladas
- Análisis visual comparativo
- Aprendizaje multimodal interactivo

### Investigación
- Análisis de corpus visuales
- Estudios de similitud semántica
- Evaluación de modelos multimodales
- Benchmarking de sistemas RAG

### Aplicaciones Comerciales
- Búsqueda visual en catálogos
- Recomendaciones basadas en imágenes
- Análisis de contenido multimedia
- Asistentes virtuales multimodales

## Limitaciones Conocidas

### Técnicas
- **Corpus Limitado**: Muestra representativa para demostración
- **Idioma**: Optimizado para español/inglés
- **Calidad de Imagen**: Sensible a imágenes de baja resolución
- **Conceptos Específicos**: Limitado a categorías del corpus

### Computacionales
- **Memoria RAM**: Mínimo 8GB recomendado
- **Tiempo de Inicialización**: ~30 segundos primera vez
- **Concurrencia**: Diseñado para uso individual


## Métricas de Corpus

### Estadísticas Flickr8k
- **Imágenes procesadas**: 1,500 (de 8,000 totales)
- **Captions por imagen**: 5 promedio
- **Vocabulario único**: ~8,000 palabras
- **Categorías principales**: Personas (40%), Animales (25%), Objetos (20%), Paisajes (15%)

### Estadísticas English Dictionary
- **Conceptos totales**: 42,052
- **Definiciones válidas**: Filtradas por longitud mínima
- **Categorías**: Objetos, animales, conceptos abstractos, lugares
- **Idioma**: Principalmente inglés con soporte multilingüe


## Referencias Académicas

1. Radford, A., et al. (2021). "Learning Transferable Visual Representations from Natural Language Supervision." ICML.

2. Johnson, J., et al. (2017). "FAISS: A Library for Efficient Similarity Search." Facebook AI Research.

3. Karpathy, A., & Fei-Fei, L. (2015). "Deep Visual-Semantic Alignments for Generating Image Descriptions." CVPR.

4. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS.


## Licencia

Este proyecto es desarrollado con fines académicos para la Escuela Politécnica Nacional. 

**Uso Académico**: Permitido con atribución apropiada  
**Uso Comercial**: Requiere autorización de los autores  
**Modificaciones**: Permitidas bajo misma licencia

---

**Fecha de Última Actualización**: Enero 2025  
**Versión**: 1.0.0  
**Estado**: Proyecto Académico Completado