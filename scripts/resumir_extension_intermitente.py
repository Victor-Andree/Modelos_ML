"""Regenera salidas descriptivas desde predicciones congeladas; sin entrenamiento."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.informe_extension import resumir_extension
if __name__=='__main__':resumir_extension(ROOT)
