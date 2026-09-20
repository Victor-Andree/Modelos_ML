"""Ejecuta una extensión nueva. No sobrescribe predicciones ni vuelve a consultar TEST."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.extension_intermitente import ejecutar
if __name__=='__main__':ejecutar(ROOT)
