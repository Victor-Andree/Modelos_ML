"""Verifica evidencia generada; ejecutar tras 06, 07 y 08."""
import json
import sys
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing.demanda_semanal import (
    construir_semanal, agregar_features, excluir_semana_corte, auditar_target, NUMERICAS)
from src.Modelos.ArimaPredictor import ArimaPredictor


def validar():
    salida = ROOT / 'resultados/semanal'
    ruta = ROOT / 'datasets/dataset_maestro_dashboard.csv'
    raw = pd.read_csv(ruta, parse_dates=['fecdoc'])
    semanal = construir_semanal(raw)
    pd.testing.assert_frame_equal(semanal, ArimaPredictor(raw).preparar_semanal())
    assert np.isclose(semanal.cantidad_neta_original.sum(), raw.cantidad.sum())
    np.testing.assert_array_equal(semanal.cantidad, semanal.cantidad_neta_original.clip(lower=0))
    guardado = pd.read_csv(salida/'dataset_semanal.csv', parse_dates=['semana'])
    pd.testing.assert_frame_equal(semanal, guardado, check_dtype=False)
    evaluable, cruzadas = excluir_semana_corte(semanal)
    pd.testing.assert_frame_equal(evaluable.reset_index(drop=True),
        pd.read_csv(salida/'dataset_semanal_evaluable.csv', parse_dates=['semana']), check_dtype=False)
    f, _ = excluir_semana_corte(agregar_features(semanal))
    listo = f.dropna(subset=NUMERICAS)
    test = listo[listo.semana.ge('2026-01-01')].sort_values(['semana','sucursal','producto'])
    assert (test.semana-pd.Timedelta(days=6)).ge(pd.Timestamp('2026-01-01')).all()
    # Comprueba directamente las fechas de las transacciones asignadas a cada semana test.
    raw['semana'] = raw.fecdoc.dt.to_period('W-SUN').dt.end_time.dt.normalize()
    participantes = raw.merge(test[['sucursal','producto','semana']], on=['sucursal','producto','semana'])
    assert participantes.fecdoc.ge('2026-01-01').all()
    ml = json.loads((salida/'experimento_ml.json').read_text(encoding='utf-8'))
    ar = json.loads((salida/'experimento_arima.json').read_text(encoding='utf-8'))
    assert ml['version_target'] == ar['version_target'] == 'neto_erp_no_negativo_corte_estricto_v1'
    assert ml['dataset_original']['sha256_dataset'] == ar['sha256_dataset'] == hashlib.sha256(ruta.read_bytes()).hexdigest()
    assert ml['sha256_constructor'] == ar['sha256_constructor'] == hashlib.sha256((ROOT/'src/preprocessing/demanda_semanal.py').read_bytes()).hexdigest()
    comparacion = pd.read_csv(salida/'comparacion_modelos.csv')
    assert len(comparacion) == 4 and comparacion.Modelo.is_unique
    for _, metrica in comparacion.iterrows():
        es_arima = metrica.Modelo.startswith('ARIMA')
        archivo = 'predicciones_arima_concatenadas.csv' if es_arima else f'predicciones_{metrica.Modelo.replace(" ", "_")}_semanal.csv'
        d = pd.read_csv(salida/archivo, parse_dates=['semana'])
        assert len(d) == metrica.Observaciones_evaluadas
        assert not d.duplicated(['sucursal','producto','semana']).any()
        assert (d.semana-pd.Timedelta(days=6)).ge(pd.Timestamp('2026-01-01')).all()
        expected = evaluable[['sucursal','producto','semana','cantidad']]
        cruce = d.merge(expected, on=['sucursal','producto','semana'], validate='one_to_one')
        assert len(cruce) == len(d)
        np.testing.assert_allclose(cruce.y_real, cruce.cantidad, rtol=0, atol=1e-12)
        if not es_arima:
            pd.testing.assert_frame_equal(d[['sucursal','producto','semana']].reset_index(drop=True),
                test[['sucursal','producto','semana']].reset_index(drop=True))
        else:
            horizonte = (d.semana-pd.to_datetime(d.origen)).dt.days // 7
            np.testing.assert_array_equal(horizonte, d.horizonte_semanas)
        error = d.y_real-d.y_pred
        np.testing.assert_allclose(error, d.error, atol=1e-12)
        np.testing.assert_allclose(abs(error), d.error_absoluto, atol=1e-12)
        esperado = [abs(error).mean(), np.sqrt((error**2).mean()),
                    1-(error**2).sum()/((d.y_real-d.y_real.mean())**2).sum()]
        np.testing.assert_allclose(esperado, [metrica.MAE,metrica.RMSE,metrica.R2], rtol=1e-10)
    pd.testing.assert_frame_equal(auditar_target(raw), pd.read_csv(salida/'auditoria_target.csv'), check_dtype=False)
    ajustes = pd.read_csv(salida/'ajustes_negativos_semanales.csv')
    assert len(ajustes) == semanal.cantidad_neta_original.lt(0).sum()
    assert ajustes.valor_antes.lt(0).all() and ajustes.valor_despues.eq(0).all()
    for nombre in ['06_Entrenamiento_ARIMA.ipynb','07_Entrenamiento_MachineLearning.ipynb','08_EDA_Intermitencia_Series.ipynb']:
        nb = nbformat.read(ROOT/'notebooks'/nombre, as_version=4)
        nbformat.validate(nb)
        for c in nb.cells:
            if c.cell_type == 'code':
                assert c.execution_count is not None
                assert not any(o.output_type == 'error' for o in c.outputs)
    print('OK: target, corte desde transacciones, paridad ARIMA/ML, test ML idéntico, horizontes ARIMA, métricas desde predicciones, auditorías, manifiestos y notebooks ejecutados.')


if __name__ == '__main__':
    validar()
