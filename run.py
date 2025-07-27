"""
Script de ejecución para ProyectoRI_2BIM
"""
import sys
import subprocess
from pathlib import Path

def main():
    """Ejecuta la aplicación"""
    
    print("🚀 Iniciando ProyectoRI_2BIM...")
    
    # Configurar path
    project_root = Path(__file__).parent
    src_dir = project_root / "src"
    sys.path.insert(0, str(src_dir))
    
    # Verificar que app.py existe
    app_file = project_root / "app.py"
    if not app_file.exists():
        print("❌ app.py no encontrado")
        return 1
    
    # Intentar ejecutar con streamlit
    try:
        subprocess.run(["streamlit", "run", str(app_file)], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.run(["python", "-m", "streamlit", "run", str(app_file)], check=True)
        except Exception as e:
            print(f"❌ Error ejecutando aplicación: {e}")
            print("Intente manualmente: streamlit run app.py")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
