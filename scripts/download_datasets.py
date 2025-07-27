"""
Script para descargar los datasets de los corpus Flickr8k y English Dictionary
"""
import sys
import os
from pathlib import Path

# Agregar src al path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.data_processing.corpus_loader import CorpusLoader
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
def setup_kaggle_credentials():
    """Guía para configurar credenciales de Kaggle"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_file = kaggle_dir / "kaggle.json"
    
    print("Configuración de Kaggle API")
    print("=" * 50)
    print()
    if not kaggle_file.exists():
        print("Configurar .env:")
        print("KAGGLE_USERNAME=username")
        print("KAGGLE_KEY=key")
        print("=" * 50)
        return True  # kagglehub no requiere credenciales obligatorias
    return True
def download_all_datasets():
    """Descarga todos los datasets necesarios"""
    print("Descargando datasets")
    print("=" * 60)
    # Verificar credenciales de Kaggle
    if not setup_kaggle_credentials():
        print("Configurar las credenciales de Kaggle para continuar")
        return False
    # Inicializar cargador
    loader = CorpusLoader()
    # Descargar Flickr8k
    print("\n Descargando corpus Flickr8k desde Kaggle")
    flickr_success = loader.download_flickr8k()
    if flickr_success:
        print("✅ Flickr8k descargado exitosamente")
    else:
        print("❌ Error descargando Flickr8k")
    # Descargar English Dictionary
    print("\nDescargando corpus English Dictionary desde Kaggle")
    dict_success = loader.download_english_dictionary()
    
    if dict_success:
        print("✅ English Dictionary descargado exitosamente")
    else:
        print("❌ Error descargando English Dictionary")   
    # Verificar descarga
    try:
        flickr_data = loader.load_flickr8k_real()
        dict_data = loader.load_english_dictionary_real()
        
        print(f"✅ Flickr8k: {len(flickr_data['images'])} imágenes cargadas")
        print(f"✅ English Dictionary: {len(dict_data)} palabras cargadas")
        
        return True
    except Exception as e:
        print(f"❌ Error verificando datasets: {e}")
        return False
def main():
    """Función principal"""
    success = download_all_datasets()
    
    print("\n" + "=" * 60)
    
    if success:
        print("-- Datasets cargados --")
        print("Ejecutar con : streamlit run app.py")
    else:
        print("Algunos datasets no se pudieron descargar")
    return 0 if success else 1
if __name__ == "__main__":
    sys.exit(main())
