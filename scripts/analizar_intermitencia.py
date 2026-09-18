"""Análisis adicional sin entrenar; ejecutar desde cualquier directorio."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.analisis_intermitencia import ejecutar
if __name__=='__main__':
    tablas=ejecutar(ROOT)
    print(tablas['distribucion_intermitencia.csv'].to_string(index=False))
    print(tablas['evaluacion_despliegue.csv'].to_string(index=False))
    print('Análisis terminado sin reentrenar ni modificar las predicciones.')
