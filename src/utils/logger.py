"""
Sistema de logging para el proyecto
"""
import logging
import sys
from pathlib import Path

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configura y retorna un logger personalizado"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Evitar añadir múltiples handlers si ya existen
    if not logger.handlers:
        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level) # Asegurar que el handler también tenga el nivel correcto
        
        # Formato del log
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    return logger
