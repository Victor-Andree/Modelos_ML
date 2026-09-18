# Reproducir la evaluación común

Con el entorno descrito por requirements-semanal.txt, desde la raíz:

```powershell
python scripts/ejecutar_experimento_semanal.py
python -m unittest test_demanda_semanal test_comparacion_comun
python scripts/validar_evidencia_semanal.py
```

La tabla vigente es resultados/semanal/comparacion_modelos_comun.csv. Los cuatro modelos comparten 3444 observaciones de 129 series. Las métricas propias están separadas en metricas_cobertura_propia.csv. La población es idéntica; los protocolos conservados de ARIMA y ML difieren.

El RAW se conserva. Las salidas anteriores de la rama principal se archivaron en resultados/historico_neto_v1; no deben mezclarse con el target depurado v2. El dashboard usa los modelos nuevos. No se modificó la Discusión del artículo.

Los manifiestos incluyen hashes exactos de bytes. .gitattributes evita que Git convierta los finales de línea de código y datos al descargarlos en otro sistema operativo.
