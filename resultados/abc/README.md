# ABC sobre demanda depurada

Ejecutar todas las celdas de `notebooks/02_Analisis_ABC_Inventarios.ipynb` para regenerar estas tablas y la auditoría. El notebook funciona desde la raíz del repositorio o desde `notebooks/`.

Se lee `datasets/dataset_maestro_dashboard.csv` y se aplica la misma función `depurar_demanda` del experimento. Se comprueba igualdad de todas las filas y columnas con `resultados/semanal/dataset_analitico_depurado.csv`. No se imputan cantidades ni se eliminan ceros: se excluyen 74 negativos (suma -93), y no existen cantidades nulas en esta fuente. Registros: 44389 → 44315; volumen: 3166 → 3259, calculado directamente de los datos.

Se ordena por cantidad descendente (producto ascendente para empates). Los límites se aplican al porcentaje acumulado sin redondear: A <= 80%, B > 80% y <= 95%, C > 95%. El producto que cruza un límite queda en la siguiente categoría.

| Categoría | Productos | Cantidad | % productos | % volumen | % volumen acumulado |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 2 | 2351 | 14.29 | 72.14 | 72.14 |
| B | 4 | 702 | 28.57 | 21.54 | 93.68 |
| C | 8 | 206 | 57.14 | 6.32 | 100.00 |
| Total | 14 | 3259 | 100.00 | 100.00 | 100.00 |

`resumen_abc_depurado.csv` conserva precisión completa de porcentajes e incluye Total. `detalle_abc_depurado.csv` contiene los 14 medicamentos con cantidades enteras, porcentajes y categoría. `auditoria_abc.json` registra exclusiones, volumen, ceros conservados y hashes de las fuentes y función de limpieza. Los dos decimales se aplican únicamente a la presentación.

No se alteran modelos, población común, métricas, predicciones ni datasets. Los manifiestos históricos de intermitencia, extensión y significancia se conservan como instantáneas: incluyen el hash anterior del notebook ABC, por lo que sus validadores de integridad completa señalarán este cambio autorizado. No se deben regenerar resultados predictivos para resolver esa diferencia histórica. Se comprobó que los otros 210 archivos previamente versionados permanecen idénticos byte a byte.
