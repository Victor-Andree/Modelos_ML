from pathlib import Path
import json
import hashlib
import joblib
import numpy as np

# Ejecutar después del entrenamiento de los tres modelos y de crear tabla_final.
nombre_modelo = 'Regresión Lineal'
tabla = tabla_final.set_index('Modelo')
if nombre_modelo not in pipeline.modelos:
    raise RuntimeError('Primero ejecuta las celdas de entrenamiento del notebook 07.')
if not (tabla.loc[nombre_modelo, 'RMSE'] <= tabla['RMSE'].min()
        and tabla.loc[nombre_modelo, 'R2'] >= tabla['R2'].max()):
    raise ValueError('Esta ejecución no confirma la selección por RMSE y R2; revisar la tabla antes de exportar.')

# Pipeline completo: One-Hot Encoder ajustado en TRAIN + LinearRegression.
# No se vuelve a entrenar con el test ni se sobrescribe el XGBoost anterior.
modelo_exportado = pipeline.modelos[nombre_modelo]
ruta_modelo = Path('modelo_regresion_lineal_semanal.pkl')
joblib.dump(modelo_exportado, ruta_modelo)

# Comprobar que guardar/cargar no cambia ninguna predicción del holdout.
modelo_cargado = joblib.load(ruta_modelo)
np.testing.assert_allclose(
    modelo_cargado.predict(X_test), pipeline.predicciones[nombre_modelo]['y_pred'].to_numpy(),
    rtol=0, atol=0)

metadata = {
    'modelo': nombre_modelo,
    'criterio_seleccion': 'Menor RMSE entre ML, respaldado por mayor R2; XGBoost tiene menor MAE en la ejecución verificada.',
    'entrenado_solo_en_train': True,
    'train_inicio': str(df_train.semana.min()),
    'train_fin': str(df_train.semana.max()),
    'fecha_corte': fecha_corte,
    'target': 'Cantidad neta semanal ERP truncada a cero después de sumar; sin normalización',
    'version_target': manifiesto['version_target'],
    'frecuencia': 'W-SUN',
    'protocolo': 'Secuencial de una semana; requiere ocho semanas previas observadas',
    'columnas_entrada': X_train.columns.tolist(),
    'tipos_entrada': {col: str(tipo) for col, tipo in X_train.dtypes.items()},
    'metricas_holdout': {k: float(tabla.loc[nombre_modelo, k]) for k in ['MAE','RMSE','R2']},
    'observaciones_holdout': len(X_test),
    'sha256_dataset': resumen_original['sha256_dataset'],
    'sha256_constructor': manifiesto['sha256_constructor'],
    'sha256_modelo': hashlib.sha256(ruta_modelo.read_bytes()).hexdigest(),
    'python': manifiesto['python'],
    'versiones': manifiesto['versiones'],
    'predicciones_negativas': 'Se conservan; no se aplica recorte posterior',
    'integracion': 'Crear features con demanda_semanal.py sobre historia semanal completa; pasar sucursal, producto y NUMERICAS. No pasar cantidad_neta_original, cantidad, total o unidades como predictores.'
}
ruta_metadata = ruta_modelo.with_suffix('.json')
ruta_metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
print('Pipeline guardado:', ruta_modelo.resolve())
print('Metadatos guardados:', ruta_metadata.resolve())
print('Verificación correcta: las predicciones recargadas coinciden exactamente con el holdout.')
