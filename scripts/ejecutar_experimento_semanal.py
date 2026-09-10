"""Ejecuta 06, 07 y 08 desde cualquier directorio, en kernels independientes."""
import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
# El entorno con las dependencias y su kernel deben corresponder al Python del comando.
os.environ.setdefault('JUPYTER_PATH', str(Path(sys.prefix) / 'share' / 'jupyter'))
nombres = sys.argv[1:] or ['06_Entrenamiento_ARIMA.ipynb',
                           '07_Entrenamiento_MachineLearning.ipynb',
                           '08_EDA_Intermitencia_Series.ipynb']
for nombre in nombres:
    ruta = ROOT / 'notebooks' / nombre
    nb = nbformat.read(ruta, as_version=4)
    print(f'Ejecutando {nombre}', flush=True)
    NotebookClient(nb, timeout=14400, kernel_name='python3',
                   resources={'metadata': {'path': str(ROOT / 'notebooks')}}).execute()
    nbformat.write(nb, ruta)
    print(f'Guardado {nombre}', flush=True)
