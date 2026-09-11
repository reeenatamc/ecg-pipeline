# Umbrales de ECGFounder derivados localmente, fold 10 de PTB-XL

Corrida local en CPU (Mac Intel, 16 GB), bajo `nice -n 10`, como respaldo mientras no
llega la corrida completa en Colab (`scripts/derive_thresholds_colab.ipynb`). Mismo
procedimiento que el notebook: z-score global, pasabanda opcional (notch 50 Hz Q=30 +
Butterworth orden 4, 0.67-40 Hz), inferencia por lotes con
`build_12lead_model`/`build_1lead_model`, y `optimal_threshold` = barrido de balanced
accuracy 0.01 a 0.99 con MIN_POSITIVES=20.

**Aviso importante: esto es solo strat_fold 10, unos 2198 registros, una decima parte
de PTB-XL.** Los 21799 registros completos del notebook de Colab dan umbrales mas
estables y muchas mas clases por encima de MIN_POSITIVES=20 (ver conclusion c). Si esa
corrida llega, sus archivos reemplazan a los de aqui: mismo procedimiento, mas datos.

Script: `derive.py`. Datos: `ptbxl_database.csv` (PhysioNet 1.0.3) y `ptbxl_label.csv`
(ECGFounder, etiquetas de 150 clases ya mapeadas a PTB-XL). WFDB leido a mano (16 bits,
little endian, intercalado por canal), sin la libreria `wfdb`: no es importable en el
venv del digitalizador sin pandas, y no se toco ese venv mas alla de instalar y luego
desinstalar `wfdb --no-deps` para comprobarlo.

## Tiempos medidos

- Descarga de 2198 registros (.hea + .dat), 8 workers: 767 s (~13 min), 0 errores.
- Prueba de 100 registros, 1lead sobre II, sin filtro: 2.6 s/100 registros.
- Prueba de 100 registros, 12lead con pasabanda (el caso mas caro): 4.6 s/100
  registros, proyectado a 1.7 min para los 2198 registros completos.
- Como 1.7 min esta muy por debajo del limite de 45 min, los runs 5 a 7 (12lead,
  12lead_bandpass, 1lead_II_bandpass) corrieron sobre el subconjunto completo, sin
  recortar a la mitad.
- Los 7 runs de inferencia completos: aproximadamente 12 minutos en total (61 a 149 s
  cada uno, mas lento cuando hay pasabanda o cuando la maquina tenia otros procesos
  activos).

## Resumen por variante

| variante | clases con umbral | AUROC media | AUROC mediana | descripcion |
|---|---:|---:|---:|---|
| 1lead_II | 23 | 0.7499 | 0.7644 | 1 derivacion (II), z-score, sin filtro |
| 1lead_I | 23 | 0.8039 | 0.8032 | 1 derivacion (I), z-score, sin filtro |
| 1lead_V1 | 23 | 0.6944 | 0.6603 | 1 derivacion (V1), z-score, sin filtro |
| 1lead_V5 | 23 | 0.7701 | 0.8102 | 1 derivacion (V5), z-score, sin filtro |
| 12lead | 23 | 0.8610 | 0.9010 | 12 derivaciones, z-score, sin filtro |
| 12lead_bandpass | 23 | 0.8674 | 0.9014 | 12 derivaciones, notch 50 + 0.67-40 Hz |
| 1lead_II_bandpass | 23 | 0.7476 | 0.7872 | 1 derivacion (II), notch 50 + 0.67-40 Hz |

## AUROC en las clases clave, por variante

"-" significa sin umbral: menos de 20 positivos en este fold, o cero positivos.

| clase | positivos (fold10) | 1lead_II | 1lead_I | 1lead_V1 | 1lead_V5 | 12lead | 12lead_bp | 1lead_II_bp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SINUS RHYTHM | 1674 | 0.8174 | 0.8341 | 0.8315 | 0.8214 | 0.7989 | 0.7972 | 0.8183 |
| NORMAL SINUS RHYTHM | 0 | - | - | - | - | - | - | - |
| ATRIAL FIBRILLATION | 152 | 0.9822 | 0.9790 | 0.9801 | 0.9822 | 0.9756 | 0.9754 | 0.9754 |
| SINUS TACHYCARDIA | 82 | 0.9922 | 0.9923 | 0.9873 | 0.9897 | 0.9837 | 0.9872 | 0.9818 |
| SINUS BRADYCARDIA | 64 | 0.9522 | 0.9465 | 0.9289 | 0.9480 | 0.9403 | 0.9478 | 0.9472 |
| NORMAL ECG | 963 | 0.8013 | 0.7903 | 0.6722 | 0.8113 | 0.8645 | 0.8692 | 0.7975 |
| ABNORMAL ECG | 0 | - | - | - | - | - | - | - |
| RIGHT BUNDLE BRANCH BLOCK | 166 | 0.8189 | 0.8400 | 0.5342 | 0.8102 | 0.9746 | 0.9777 | 0.8160 |
| LEFT BUNDLE BRANCH BLOCK | 62 | 0.9593 | 0.9825 | 0.9596 | 0.9803 | 0.9837 | 0.9819 | 0.9493 |
| ATRIAL FLUTTER | 7 | - | - | - | - | - | - | - |
| 1ST DEGREE AV BLOCK* | 79 | 0.8303 | 0.9155 | 0.7711 | 0.8201 | 0.9054 | 0.9027 | 0.8564 |
| PREMATURE VENTRICULAR COMPLEXES | 114 | 0.9086 | 0.8772 | 0.9484 | 0.9379 | 0.9795 | 0.9838 | 0.9338 |

\* La lista de 150 clases de ECGFounder no tiene una clase llamada exactamente
"1ST DEGREE AV BLOCK": la mas cercana es "WITH 1ST DEGREE AV BLOCK" (indice 80 de
`tasks.txt`), y es la que se reporta aqui.

NORMAL SINUS RHYTHM y ABNORMAL ECG tienen cero positivos en las etiquetas de
ECGFounder para PTB-XL (no es un problema del fold, se repite en todo el dataset
segun `ptbxl_label.csv`): esas dos clases simplemente no se pueden evaluar con este
mapeo de etiquetas. ATRIAL FLUTTER tiene 7 positivos en fold10, por debajo de
MIN_POSITIVES=20, por eso tampoco recibe umbral aqui (en el set completo, con ~10x
mas registros, es esperable que si lo alcance).

## Umbrales de 12lead para las clases clave (lo que iria en --thresholds)

| clase | positivos | umbral | AUROC |
|---|---:|---:|---:|
| SINUS RHYTHM | 1674 | 0.98 | 0.7989 |
| SINUS BRADYCARDIA | 64 | 0.94 | 0.9403 |
| SINUS TACHYCARDIA | 82 | 0.92 | 0.9837 |
| ATRIAL FIBRILLATION | 152 | 0.40 | 0.9756 |
| NORMAL ECG | 963 | 0.03 | 0.8645 |
| RIGHT BUNDLE BRANCH BLOCK | 166 | 0.17 | 0.9746 |
| LEFT BUNDLE BRANCH BLOCK | 62 | 0.14 | 0.9837 |
| WITH 1ST DEGREE AV BLOCK | 79 | 0.01 | 0.9054 |
| PREMATURE VENTRICULAR COMPLEXES | 114 | 0.24 | 0.9795 |

## Las tres conclusiones que piden los numeros

**a. El pasabanda ayuda, pero apenas, y no de forma pareja entre modelos.**
En el modelo de 12 derivaciones, `12lead_bandpass` supera a `12lead` en AUROC media
(0.8674 contra 0.8610, +0.0064) y en mediana (0.9014 contra 0.9010, practicamente
igual). En las clases clave el pasabanda mejora RIGHT BUNDLE BRANCH BLOCK (0.9746 a
0.9777), PREMATURE VENTRICULAR COMPLEXES (0.9795 a 0.9838) y NORMAL ECG (0.8645 a
0.8692), pero empeora levemente SINUS RHYTHM (0.7989 a 0.7972). En el modelo de 1
derivacion pasa lo contrario: `1lead_II_bandpass` queda por debajo de `1lead_II` en
AUROC media (0.7476 contra 0.7499) y pierde en casi todas las clases clave (AFIB,
taquicardia, bradicardia, NORMAL ECG, RBBB, LBBB), con la excepcion de PVC. La
diferencia de 0.0064 en el modelo de 12 derivaciones es del tamano del ruido
esperable con una decima parte del dataset, y el filtro directamente no ayuda al
checkpoint de 1 derivacion. Con estos numeros no hay caso para implementar el
pasabanda por defecto; el punto abierto del NOTICE deberia esperar a la corrida
completa en Colab antes de cerrarse en cualquier direccion.

**b. El checkpoint de 1 derivacion rinde mejor en I (para lo que fue afinado) y peor
en V1; II queda en un punto intermedio razonable.** Orden por AUROC media:
1lead_I (0.8039) > 1lead_V5 (0.7701) > 1lead_II (0.7499) > 1lead_V1 (0.6944). La
brecha entre I y II es de unos 5 puntos de AUROC media, no trivial pero tampoco
descalificante: II sigue siendo una eleccion razonable para el pathway de ritmo,
que es exactamente lo que hace el pipeline. El caso mas claro para actuar es
RIGHT BUNDLE BRANCH BLOCK: 0.9746 en 12lead, 0.8189-0.8400 en II/I, pero solo 0.5342
en V1, practicamente el azar. Esto confirma en numeros lo que ya advertia el
comentario en `interpret_ecg.py`: si V1 cae mucho, conviene sacarla de
`PREFERRED_RHYTHM_LEADS` o ponderarla menos, al menos para clases de conduccion como
BBB. V5 sorprende por lo bien que le va (mediana 0.8102, la mas alta de las cuatro),
casi a la par de I.

**c. Solo 23 de las 150 clases reciben umbral en cualquier variante, y es un limite
de este fold, no del metodo.** El corte MIN_POSITIVES=20 deja fuera a la enorme
mayoria de las 150 clases porque 2198 registros no alcanzan para juntar 20 positivos
en las clases menos frecuentes (ademas de NORMAL SINUS RHYTHM y ABNORMAL ECG, que
tienen cero positivos en todo PTB-XL segun este mapeo de etiquetas, no solo en este
fold). El numero de 23 es identico en las 7 variantes porque las 2198 etiquetas son
las mismas en todos los runs (ninguno se recorto a la mitad). Con los 21799 registros
completos del notebook de Colab, casi 10 veces mas datos, es esperable que bastantes
mas clases crucen el umbral de 20 positivos, incluida ATRIAL FLUTTER (7 positivos
aqui, probablemente mas de 20 en el dataset completo). Esta es la razon principal por
la que estos archivos son un respaldo y no un reemplazo de la corrida completa.

## Archivos en este directorio

- `thresholds_<variante>.json`: umbral por clase, solo las 23 clases que llegan a
  MIN_POSITIVES=20 en este fold.
- `metrics.csv`: AUROC, positivos, umbral y balanced accuracy por clase y variante,
  las 150 clases x 7 variantes (con AUROC/umbral vacios donde no hay suficientes
  positivos).
- `probs_<variante>.npy`: probabilidades crudas (2198 x 150), por si hace falta
  recalcular umbrales con otro criterio sin volver a correr los modelos.
- `run_manifest.json` / `run_info.json`: que registros y que tiempos corresponden a
  cada variante, para reproducir o auditar la corrida.
