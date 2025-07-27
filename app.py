"""
Punto de entrada principal para la aplicación Streamlit
"""
import sys
import os
from pathlib import Path

# Configurar el path para importaciones
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
# Agregar directorios al path de Python
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(src_dir))
# Configurar variable de entorno para módulos
os.environ['PYTHONPATH'] = str(src_dir)
# Importar y ejecutar la aplicación principal
if __name__ == "__main__":
    try:
        from src.main import main
        main()
    except ImportError as e:
        print(f"Error de importación: {e}")
        # Importación alternativa
        try:
            import main
            main.main()
        except ImportError:
            print("Error: No se puede importar el módulo principal")
            sys.exit(1)
