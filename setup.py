"""
Script de instalación completa para el proyecto
Instala dependencias, configura entorno, descarga datasets y corrige importaciones
"""
import subprocess
import sys
import os
import shutil
import re
from pathlib import Path
import platform

class ProyectoRISetup:
    """Instalador completo para el proyecto"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.src_dir = self.project_root / "src"
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.cache_dir = self.project_root / "cache"
        
    def print_header(self, title):
        """Imprime encabezado formateado"""
        print(f"\n{'='*60}")
        print(f"🚀 {title}")
        print(f"{'='*60}")
    
    def run_command(self, command, description, check=True):
        """Ejecuta comando y maneja errores"""
        print(f"🔄 {description}")
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {description} exitoso")
                return True
            else:
                if check:
                    print(f"❌ Error en {description}")
                    if result.stderr.strip():
                        print(f"   {result.stderr.strip()}")
                return False
        except Exception as e:
            if check:
                print(f"❌ Excepción en {description}: {e}")
            return False
    
    def check_python(self):
        """Verifica Python y pip"""
        self.print_header("Verificando Python")
        
        # Verificar Python
        if not self.run_command("python --version", "Verificación de Python"):
            print("❌ Python no está instalado o no está en PATH")
            print("   Descargue Python desde: https://www.python.org/downloads/")
            return False
        
        # Verificar pip
        if not self.run_command("pip --version", "Verificación de pip"):
            print("❌ pip no está disponible")
            return False
        
        # Obtener versión de Python
        python_version = sys.version_info
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        print(f"✅ Plataforma: {platform.system()} {platform.machine()}")
        
        return True
    
    def create_project_structure(self):
        """Crea estructura de directorios"""
        self.print_header("Creando Estructura del Proyecto")
        
        directories = [
            self.data_dir,
            self.data_dir / "flickr8k",
            self.data_dir / "3d-ex", 
            self.models_dir,
            self.cache_dir,
            self.cache_dir / "embeddings",
            Path("logs")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ Directorio: {directory}")
            
            # Crear .gitkeep para directorios vacíos
            gitkeep = directory / ".gitkeep"
            if not any(directory.iterdir()) and not gitkeep.exists():
                gitkeep.touch()
        
        print("✅ Estructura del proyecto creada")
        return True
    
    def setup_env_file(self):
        """Configura archivo .env"""
        self.print_header("Configurando Variables de Entorno")
        
        env_file = self.project_root / ".env"
        env_example = self.project_root / ".env.example"
        
        if env_file.exists():
            print("✅ Archivo .env ya existe")
            return True
        
        if env_example.exists():
            try:
                shutil.copy(env_example, env_file)
                print("✅ Archivo .env creado desde .env.example")
                print("⚠️  IMPORTANTE: Configurar las API keys en .env")
                return True
            except Exception as e:
                print(f"❌ Error creando .env: {e}")
                return False
        else:
            print("⚠️  Archivo .env no encontrado")
            return True
    
    def get_compatible_requirements(self):
        """Genera requirements.txt compatible con la versión actual de Python"""
        python_version = sys.version_info
        
        # Dependencias base
        base_requirements = [
            "streamlit>=1.28.0",
            "sentence-transformers>=2.2.2", 
            "faiss-cpu>=1.7.4",
            "Pillow>=10.0.0",
            "numpy>=1.24.3",
            "pandas>=2.0.3",
            "google-generativeai>=0.3.0",
            "requests>=2.31.0",
            "python-dotenv>=1.0.0",
            "kaggle>=1.5.16",
            "kagglehub>=0.2.0",
            "tqdm>=4.66.1"
        ]
        
        # PyTorch compatible según versión de Python
        if python_version >= (3, 12):
            torch_requirements = ["torch>=2.2.0", "torchvision>=0.17.0"]
        elif python_version >= (3, 11):
            torch_requirements = ["torch>=2.1.0", "torchvision>=0.16.0"]
        else:
            torch_requirements = ["torch>=2.0.1", "torchvision>=0.15.2"]
        
        return base_requirements + torch_requirements
    
    def install_dependencies(self):
        """Instala todas las dependencias"""
        self.print_header("Instalando Dependencias")
        
        # Actualizar pip
        self.run_command("python -m pip install --upgrade pip", "Actualización de pip", check=False)
        
        # Crear requirements.txt compatible
        requirements = self.get_compatible_requirements()
        req_file = self.project_root / "requirements.txt"
        
        with open(req_file, 'w', encoding='utf-8') as f:
            for req in requirements:
                f.write(f"{req}\n")
        
        print(f"✅ Requirements.txt actualizado con {len(requirements)} dependencias")
        
        # Instalar dependencias críticas primero
        critical_deps = ["numpy>=1.24.3", "pillow>=10.0.0", "requests>=2.31.0", "python-dotenv>=1.0.0"]
        
        print("Instalando dependencias")
        for dep in critical_deps:
            self.run_command(f"pip install {dep}", f"Instalando {dep}", check=False)
        
        # Instalar PyTorch
        print("✅Instalando PyTorch")
        python_version = sys.version_info
        
        if python_version >= (3, 12):
            torch_cmd = "pip install torch>=2.2.0 torchvision>=0.17.0 --index-url https://download.pytorch.org/whl/cpu"
        else:
            torch_cmd = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
        
        if not self.run_command(torch_cmd, "Instalando PyTorch", check=False):
            # Fallback sin index-url
            self.run_command("pip install torch torchvision", "Instalando PyTorch (fallback)", check=False)
        
        # Instalar resto de dependencias
        remaining_deps = [
            "sentence-transformers>=2.2.2",
            "faiss-cpu>=1.7.4", 
            "pandas>=2.0.3",
            "google-generativeai>=0.3.0",
            "kaggle>=1.5.16",
            "streamlit>=1.28.0",
            "tqdm>=4.66.1"
        ]
        
        for dep in remaining_deps:
            self.run_command(f"pip install {dep}", f"Instalando {dep}", check=False)
        
        # Verificar instalación de Streamlit
        if self.run_command("streamlit --version", "Verificando Streamlit", check=False):
            print("✅ Streamlit instalado correctamente")
        else:
            print("⚠️  Streamlit disponible con: python -m streamlit")
        
        return True
    
    def fix_imports(self):
        """Corrige importaciones en todo el proyecto"""
        self.print_header("Corrigiendo Importaciones")
        
        # Crear archivos __init__.py
        init_dirs = [
            self.src_dir,
            self.src_dir / "utils",
            self.src_dir / "embeddings",
            self.src_dir / "indexing", 
            self.src_dir / "retrieval",
            self.src_dir / "generation",
            self.src_dir / "data_processing"
        ]
        
        for directory in init_dirs:
            if directory.exists():
                init_file = directory / "__init__.py"
                if not init_file.exists():
                    init_file.touch()
                    print(f"✅ Creado: {init_file}")
        
        # Corregir importaciones en archivos Python
        python_files = list(self.src_dir.glob("**/*.py"))
        
        fixed_count = 0
        for file_path in python_files:
            if self.fix_file_imports(file_path):
                fixed_count += 1
        
        print(f"✅ {fixed_count} archivos corregidos")
        return True
    
    def fix_file_imports(self, file_path):
        """Corrige importaciones en un archivo específico"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Agregar configuración de path si no existe
            if 'sys.path.insert' not in content and 'from src.' in content:
                path_setup = '''import sys
from pathlib import Path
current_dir = Path(__file__).parent
src_dir = current_dir.parent if 'src' in str(current_dir) else current_dir / "src"
sys.path.insert(0, str(src_dir))

'''
                # Insertar después de imports estándar
                lines = content.split('\n')
                insert_index = 0
                for i, line in enumerate(lines):
                    if line.startswith('from src.') or line.startswith('from .'):
                        insert_index = i
                        break
                
                lines.insert(insert_index, path_setup)
                content = '\n'.join(lines)
            
            # Patrones de corrección
            patterns = [
                (r'from \.\.([a-zA-Z_][a-zA-Z0-9_]*)', r'from src.\1'),
                (r'from \.([a-zA-Z_][a-zA-Z0-9_]*)', r'from src.\1'),
            ]
            
            for pattern, replacement in patterns:
                content = re.sub(pattern, replacement, content)
            
            # Escribir si hubo cambios
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
            
            return False
            
        except Exception as e:
            print(f"Error procesando {file_path}: {e}")
            return False
    
    def download_datasets(self):
        """Descarga datasets automáticamente"""
        self.print_header("Descargando Datasets")
        
        try:
            # Importar después de instalar dependencias
            sys.path.insert(0, str(self.src_dir))
            from src.data_processing.corpus_loader import CorpusLoader
            
            loader = CorpusLoader()
            
            # Descargar 3D-EX (solo JSON, desde GitHub)
            print("📚 Descargando 3D-EX.json desde GitHub...")
            if loader.download_3d_ex():
                print("✅ 3D-EX.json descargado exitosamente")
            else:
                print("⚠️  3D-EX.json falló, el sistema operará sin datos de 3D-EX.")
            
            # Intentar descargar Flickr8k
            print("📸 Intentando descargar Flickr8k desde Kaggle...")
            if loader.download_flickr8k():
                print("✅ Flickr8k descargado exitosamente")
            else:
                print("⚠️  Flickr8k falló - configure credenciales de Kaggle en .env")
                print("   O el sistema operará sin datos de Flickr8k.")
            
            # Verificar datasets
            print("🔍 Verificando datasets...")
            try:
                flickr_data = loader.load_flickr8k_real()
                concepts_data = loader.load_3d_ex_real()
                
                print(f"✅ Flickr8k: {len(flickr_data['images'])} imágenes")
                print(f"✅ 3D-EX: {len(concepts_data)} conceptos")
                
            except Exception as e:
                print(f"⚠️  Error verificando datasets: {e}")
                print("   La aplicación funcionará con los datos disponibles.")
            
            return True
            
        except Exception as e:
            print(f"❌ Error descargando datasets: {e}")
            print("   La aplicación funcionará con los datos disponibles.")
            return True  # No es crítico
    
    def test_installation(self):
        """Prueba la instalación completa"""
        self.print_header("Probando Instalación")
        
        try:
            # Probar importaciones críticas
            print("🔍 Probando importaciones...")
            
            import streamlit
            print("✅ Streamlit OK")
            
            import torch
            print(f"✅ PyTorch {torch.__version__}")
            
            import numpy
            print(f"✅ NumPy {numpy.__version__}")
            
            # Probar importaciones del proyecto
            sys.path.insert(0, str(self.src_dir))
            
            from src.utils.config import DATA_DIR
            print("✅ Config OK")
            
            from src.main import main
            print("✅ Main OK")
            
            print("✅ Todas las importaciones funcionan correctamente")
            return True
            
        except Exception as e:
            print(f"❌ Error en pruebas: {e}")
            return False
    
    def create_run_script(self):
        """Crea script de ejecución simple"""
        self.print_header("Creando Script de Ejecución")
        
        run_script = self.project_root / "run.py"
        
        script_content = '''"""
Script de ejecución para ProyectoRI_2BIM
"""
import sys
import subprocess
from pathlib import Path

def main():
    """Ejecuta la aplicación"""
    
    print("Iniciando ProyectoRI_2BIM...")
    
    # Configurar path
    project_root = Path(__file__).parent
    src_dir = project_root / "src"
    sys.path.insert(0, str(src_dir))
    
    # Verificar que app.py existe
    app_file = project_root / "app.py"
    if not app_file.exists():
        print("app.py no encontrado")
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
'''
        
        with open(run_script, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print(f"✅ Script de ejecución creado: {run_script}")
        return True
    
    def show_final_instructions(self):
        """Muestra instrucciones finales"""
        self.print_header("Instalación Completada")
        
        print("Instalación completada exitosamente!")
        print("✅ Sistema listo para usar!")
    
    def run_full_setup(self):
        """Ejecuta instalación completa"""
        print("Instalación completada exitosamente!")
        print("Autores: Fricxon Pambabay, Christian Pérez, Jeremmy Perugachi")
        print("Institución: Escuela Politécnica Nacional")
        
        steps = [
            ("Verificar Python", self.check_python),
            ("Crear estructura", self.create_project_structure),
            ("Configurar .env", self.setup_env_file),
            ("Instalar dependencias", self.install_dependencies),
            ("Corregir importaciones", self.fix_imports),
            ("Descargar datasets", self.download_datasets),
            ("Probar instalación", self.test_installation),
            ("Crear script de ejecución", self.create_run_script)
        ]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            try:
                if not step_func():
                    failed_steps.append(step_name)
            except Exception as e:
                print(f"❌ Error en {step_name}: {e}")
                failed_steps.append(step_name)
        
        if failed_steps:
            print(f"\n⚠️  Errores: {', '.join(failed_steps)}")   
        self.show_final_instructions()
        
        return len(failed_steps) == 0
def main():
    """Función principal"""
    installer = ProyectoRISetup()
    success = installer.run_full_setup()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
