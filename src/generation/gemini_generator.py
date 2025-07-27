"""
Generador de respuestas usando Gemini 2.0 Flash
"""
import google.generativeai as genai
from typing import Dict, Optional
import os
import sys
from pathlib import Path

# Configurar path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# Importaciones absolutas
try:
    from src.utils.config import GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_API_KEY
    from src.utils.logger import setup_logger
except ImportError:
    from utils.config import GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_API_KEY
    from utils.logger import setup_logger

logger = setup_logger(__name__)

class GeminiGenerator:
    """Generador de respuestas usando Gemini 2.0 Flash"""
    
    def __init__(self, api_key: Optional[str] = None):
        # Usar API key del parámetro, variable de entorno, o None
        self.api_key = api_key or GEMINI_API_KEY
        self.model = None
        self._configure_gemini()
    
    def _configure_gemini(self):
        """Configura el cliente de Gemini"""
        if not self.api_key:
            logger.error("API key de Gemini no configurada. Configurar GEMINI_API_KEY en .env")
            raise RuntimeError("GEMINI_API_KEY requerida para funcionamiento del sistema")
        
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config={
                    "temperature": GEMINI_TEMPERATURE,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 1000,
                },
                safety_settings=[
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
            )
            logger.info(f"Cliente Gemini configurado correctamente con modelo {GEMINI_MODEL}")
            logger.info("GeminiGenerator: Modelo Gemini inicializado y listo para usar.")
        except Exception as e:
            logger.error(f"Error configurando Gemini: {e}")
            raise RuntimeError(f"Error configurando Gemini: {e}")
    
    def generate_response(self, query: str, context: str, query_type: str) -> str:
        """
        Genera respuesta basada en consulta y contexto recuperado
        
        Args:
            query: Consulta original del usuario
            context: Contexto recuperado del sistema RAG
            query_type: Tipo de consulta ("image" o "text")
            
        Returns:
            Respuesta generada
            
        Raises:
            RuntimeError: Si no se puede generar respuesta
        """
        if not self.model:
            raise RuntimeError("Modelo Gemini no inicializado")
        
        # Validar que el contexto no esté vacío
        if not context or context.strip() == "":
            raise ValueError("El contexto para generación está vacío. No se encontraron resultados relevantes.")
        
        prompt = self._build_prompt(query, context, query_type)
        
        # Validar que el prompt no esté vacío
        if not prompt or prompt.strip() == "":
            raise ValueError("El prompt generado está vacío")
        
        try:
            logger.info(f"Generando respuesta con Gemini para consulta tipo: {query_type}")
            response = self.model.generate_content(prompt)
            
            # Logging detallado de la respuesta
            logger.info(f"Respuesta recibida de Gemini: {type(response)}")
            logger.info(f"Atributos de respuesta: {dir(response)}")
            
            # Verificar si hay texto en la respuesta
            if hasattr(response, 'text') and response.text:
                logger.info("Respuesta generada exitosamente")
                return response.text
            
            # Verificar si hay candidatos
            if hasattr(response, 'candidates') and response.candidates:
                logger.info(f"Número de candidatos: {len(response.candidates)}")
                for i, candidate in enumerate(response.candidates):
                    logger.info(f"Candidato {i}: {candidate}")
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            for j, part in enumerate(candidate.content.parts):
                                logger.info(f"Parte {j}: {part}")
                                if hasattr(part, 'text') and part.text:
                                    logger.info("Texto encontrado en candidato")
                                    return part.text
                    
                    # Verificar finish_reason
                    if hasattr(candidate, 'finish_reason'):
                        logger.warning(f"Finish reason: {candidate.finish_reason}")
                        if candidate.finish_reason == "SAFETY":
                            raise RuntimeError("Contenido bloqueado por filtros de seguridad de Gemini")
                        elif candidate.finish_reason == "MAX_TOKENS":
                            raise RuntimeError("Respuesta truncada por límite de tokens")
                        elif candidate.finish_reason == "RECITATION":
                            raise RuntimeError("Contenido bloqueado por políticas de recitación")
            
            # Si llegamos aquí, no hay texto disponible
            logger.error("No se encontró texto en la respuesta de Gemini")
            logger.error(f"Respuesta completa: {response}")
            raise RuntimeError("Gemini no generó respuesta de texto válida")
            
        except Exception as e:
            logger.error(f"Error generando respuesta con Gemini: {e}")
            raise RuntimeError(f"Error generando respuesta: {e}")
    
    def _build_prompt(self, query: str, context: str, query_type: str) -> str:
        """Construye prompt estructurado para Gemini"""
        
        logger.info(f"Construyendo prompt para tipo: {query_type}")
        logger.debug(f"Contexto recibido (longitud: {len(context)}): {context[:200]}...")
        
        if query_type == "image":
            prompt_template = """
**Rol:** Asistente experto en análisis visual y síntesis de información multimodal.

**Objetivo:** Analizar la imagen proporcionada y generar una respuesta informativa basada en imágenes similares y conceptos relacionados del corpus.

**Metodología:**
1. Analizar las descripciones de imágenes similares encontradas
2. Identificar patrones visuales y semánticos comunes
3. Utilizar conceptos relacionados para enriquecer la explicación
4. Proporcionar una respuesta educativa y completa

**Contexto recuperado:**
{context}

**Instrucciones específicas:**
- Inicia identificando qué se observa en la imagen
- Proporciona información detallada basada en el contexto recuperado
- Menciona elementos visuales relevantes encontrados en imágenes similares
- Incluye definiciones de conceptos cuando sea apropiado
- Mantén un tono educativo e informativo

**FORMATO DE RESPUESTA ESPERADO:**


**[concepto_identificado]**

Un/Una [concepto_identificado] es [definición completa basada en el contexto del corpus].

**FIN DE LA RESPUESTA MOSTRADA AL USUARIO**

**CASOS ESPECIALES:**
- Si la consulta contiene múltiples conceptos, enfócate en el MÁS RELEVANTE o PRINCIPAL
- Si la consulta es muy general, identifica el concepto más específico del contexto recuperado
- Si no puedes identificar un concepto claro, utiliza el término más importante de la consulta

**IMPORTANTE:** 
- SIEMPRE identifica un concepto principal, incluso si la pregunta es indirecta
- El concepto debe ser un SUSTANTIVO (objeto, animal, cosa, proceso, etc.)
- Ignora palabras como "qué", "cómo", "cuándo", "dónde" - enfócate en el TEMA central

**Si no hay información suficiente:**
"No se encontró información en el corpus"

**Análisis de la imagen:**
"""
        else:  # text query
            prompt_template = """
**Rol:** Asistente experto en recuperación de información multimodal.

**Objetivo:** Responder la consulta textual utilizando información recuperada de imágenes relacionadas y conceptos del corpus.

**Metodología:**
1. Analizar la consulta textual del usuario
2. Sintetizar información de imágenes relacionadas encontradas
3. Integrar definiciones de conceptos relevantes
4. Generar respuesta comprehensiva y educativa

**Consulta del usuario:** {query}

**Contexto recuperado:**
{context}

**Instrucciones específicas:**
- Responde directamente a la consulta del usuario
- Utiliza información de imágenes relacionadas para enriquecer la respuesta
- Incluye definiciones y explicaciones de conceptos relevantes
- Proporciona ejemplos visuales cuando sea apropiado
- Mantén coherencia entre información textual y visual

**FORMATO DE RESPUESTA ESPERADO:**

**[concepto_identificado]**

Un/Una [concepto_identificado] es [definición completa basada en el contexto del corpus]. [Continúa con toda la información disponible del contexto recuperado]...

**FIN DE LA RESPUESTA MOSTRADA AL USUARIO**

**CASOS ESPECIALES:**
- Si la consulta contiene múltiples conceptos, enfócate en el MÁS RELEVANTE o PRINCIPAL
- Si la consulta es muy general, identifica el concepto más específico del contexto recuperado
- Si no puedes identificar un concepto claro, utiliza el término más importante de la consulta

**IMPORTANTE:** 
- SIEMPRE identifica un concepto principal, incluso si la pregunta es indirecta
- El concepto debe ser un SUSTANTIVO (objeto, animal, cosa, proceso, etc.)
- Ignora palabras como "qué", "cómo", "cuándo", "dónde" - enfócate en el TEMA central

**Si no hay información suficiente:**
"No se encontró información en el corpus"

**Respuesta:**
"""
        
        formatted_prompt = prompt_template.format(
            query=query if query_type == "text" else "",
            context=context
        )
        
        logger.info(f"Prompt generado (longitud: {len(formatted_prompt)})")
        logger.debug(f"Prompt completo: {formatted_prompt[:300]}...")
        
        # Validar que el prompt formateado no esté vacío
        if not formatted_prompt or formatted_prompt.strip() == "":
            logger.error("Prompt formateado está vacío")
            raise ValueError("Error interno: prompt formateado vacío")
        
        return formatted_prompt
