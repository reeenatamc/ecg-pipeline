# Rendimiento del análisis

Medición de dónde se va el tiempo de un análisis, estudio de la resolución de trabajo del
digitalizador (el reescalado) y efecto de cada optimización. Todo medido el 12 de septiembre
de 2026 en un MacBook Pro 2019 (Intel, 8 núcleos físicos, 16 GB), en serie y con
`nice -n 10`, con la máquina compartida con el simulador y el worker de api-EKG en reposo.
El servidor de destino (8 vCPU compartidas, 16 GB) no se midió.

## 1. Perfil de una corrida

Imagen de referencia: `ejemplo-ecg-normal.png` (797x446), que el preprocesado amplía x3
con Lanczos a 2391x1338, el tamaño con el que llega un estudio típico. Se midió con un
arnés que reproduce `src.digitize.process_one_file` paso a paso y cronometra también lo que
el digitalizador no cronometra (imports, carga de pesos, identificación de layout, escritura).

| Etapa | Segundos |
|---|---:|
| Import de torch | 1,3 |
| Import del digitalizador | 1,6 |
| Carga de los dos U-Net | 1,9 |
| Decodificar imagen | 0,2 |
| Segmentación (U-Net) | 13,4 a 15,1 |
| Detección de perspectiva | 2,8 a 3,9 |
| Recorte | 1,9 a 2,0 |
| Remuestreo de mapas | 0,5 |
| Búsqueda de tamaño de píxel | 2,8 a 3,0 |
| Extracción de señal | 4,5 a 5,0 |
| Identificación de layout (segundo U-Net) | 9,5 a 12,7 |
| Guardar CSV | 0,04 |
| Guardar PNG de depuración | 5,4 a 6,0 |
| ECGFounder: carga | 0,6 |
| ECGFounder: inferencia | 0,1 |

Por imagen, unos 45 s de digitalización más unos 5 s de arranque del proceso. La corrida
completa con `python -m ecg_pipeline` (subproceso, rama main) tardó 56,9 s de pared con
5,5 GB de memoria residente pico. Los 99 s de la medición anterior se tomaron con la máquina
cargada; en reposo el mismo análisis ya baja del minuto en este Mac.

ECGFounder es irrelevante para el tiempo total. Lo que cuesta es la segmentación y la
identificación de layout, las dos redes que trabajan a resolución completa.

## 2. Estudio del reescalado

### Qué hace el redimensionado

`InferenceWrapper._resample_image` (Open-ECG-Digitizer, `src/model/inference_wrapper.py`)
reduce la imagen con `F.interpolate` bilineal y antialias solo si su lado mayor supera
`resample_size`, y la amplía si su lado menor queda por debajo de 512 px. El valor se
controla por configuración: `MODEL.KWARGS.resample_size` en `configs/digitizer_cpu.yml`,
que vale 3000. Como las imágenes que llegan miden entre 2200 y 2400 px de ancho, hoy la
segmentación trabaja a tamaño nativo: 3000 es, en la práctica, "sin reducir".

### Método

La validación del proyecto (sección 2b de la auditoría): registros de PTB-XL del fold 10
renderizados con ECG-Image-Kit en versión limpia y distorsionada, digitalizados, y
comparados por derivación contra la señal WFDB original con
`scripts/validate_against_wfdb.py` (rama `validacion`), con búsqueda de desfase de hasta 50 ms.

Las imágenes ya no estaban en disco. Se regeneraron con los mismos comandos documentados en
esa rama (ECG-Image-Kit, commit 27b90f5, semilla 42, 200 dpi, 3x4 con tira II, sin pulso de
calibración; la versión distorsionada con rotación, ruido, recorte, temperatura y arrugas).
El módulo de texto manuscrito de ECG-Image-Kit se sustituyó por uno vacío para no instalar
tensorflow: esos comandos no lo usan. Las imágenes miden 2200x1700 y no se suben al repositorio.

Métricas por registro, como en el CSV de la auditoría: mediana sobre las derivaciones de la
correlación r y del SNR. Por variante se reporta la media de esas medianas.

El SNR se calcula con la media de cada traza restada. El script, tal como está en la rama
`validacion`, calcula el SNR sin restarla, y eso castiga desplazamientos constantes de 60 a
120 µV en algunas derivaciones que no cambian la forma (la r no se mueve). Con la media
restada, la línea base reproduce el CSV de la auditoría registro a registro: SNR a 0,4 dB o
menos y r igual a tres decimales. El SNR sin restar la media se muestra también en la tabla.

Subconjunto: 12 de los 24 registros en sus dos variantes (24 imágenes por escala), elegidos
para cubrir los cinco grupos diagnósticos con al menos dos registros cada uno e incluir los
distorsionados más difíciles según la auditoría (1355, 8215, 8645, 7221, 6231 y 17386, los de
r mínima o SNR más bajos): NORM 1172, 1355, 8645; AFIB 7221, 8215, 17386; CRBBB/CLBBB 17690,
2433; IMI/AMI 6231, 18506; SBRAD/STACH 19424, 8507. El set completo a cuatro escalas son 192
digitalizaciones, unas cuatro horas en este Mac; el subconjunto fueron 96 en 64 minutos, más
unos 4 minutos de generación de imágenes.

### Tolerancia

Una escala reducida se acepta solo si, frente a la línea base medida con el mismo método:

- el SNR medio no empeora más de 0,5 dB en ninguna variante,
- la r media no empeora más de 0,002 en ninguna variante,
- no aparece ninguna compuerta nueva, ningún layout distinto y ninguna tira fuera de II.

0,5 dB y 0,002 son del orden de la diferencia entre la línea base regenerada y la auditoría
(ruido de la propia medición); cualquier caída mayor es pérdida real. Las compuertas se
exigen sin tolerancia porque una tira atribuida a otra derivación cambia qué trazo lee el modelo.

### Resultados

Tiempos en segundos por imagen, media de las 24 imágenes, 8 hilos, sin PNG de depuración.

| Escala | Tamaño de trabajo | s/imagen | Segmentación | Layout | Tamaño de píxel | RSS pico MB | SNR limpio | SNR distorsionado | r limpio | r distorsionado | Layout correcto | Tira en II | Degradados | SNR sin centrar |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3000 (actual) | 2200x1700 | 44,9 | 17,8 | 12,6 | 4,4 | 6441 | 18,42 | 15,79 | 0,9929 | 0,9870 | 24/24 | 24/24 | 0/24 | 15,90 / 13,99 |
| 2000 | 2000x1545 | 41,2 | 15,5 | 11,5 | 5,2 | 6320 | 17,90 | 14,87 | 0,9919 | 0,9840 | 24/24 | 23/24 | 0/24 | 15,70 / 13,21 |
| 1600 | 1600x1236 | 26,7 | 7,7 | 7,0 | 5,4 | 5013 | 14,80 | 12,34 | 0,9831 | 0,9690 | 24/24 | 24/24 | 0/24 | 13,33 / 11,25 |
| 1200 | 1200x927 | 39,7 | 10,4 | 7,8 | 11,7 | 3068 | 11,05 | 4,75 | 0,9598 | 0,6292 | 23/24 | 23/24 | 3/24 | 10,37 / 4,45 |

Diferencias por registro frente a la línea base (SNR en dB / r):

| Registro | Grupo | Auditoría | 3000 | 2000 | 1600 | 1200 |
|---|---|---|---|---|---|---|
| limpio 1172 | NORM | 16,9 / 0,991 | 16,9 / 0,991 | +0,7 / +0,000 | -3,1 / -0,011 | -5,9 / -0,028 |
| limpio 1355 | NORM | 16,8 / 0,990 | 16,9 / 0,990 | +0,3 / +0,001 | -3,4 / -0,012 | -7,5 / -0,049 |
| limpio 8645 | NORM | 17,2 / 0,991 | 17,3 / 0,991 | -0,5 / -0,001 | -3,6 / -0,011 | -8,2 / -0,051 |
| limpio 7221 | AFIB | 16,3 / 0,989 | 16,3 / 0,989 | -0,5 / -0,002 | -3,1 / -0,012 | -6,4 / -0,040 |
| limpio 8215 | AFIB | 17,7 / 0,992 | 17,7 / 0,992 | -1,6 / -0,004 | -4,8 / -0,017 | -7,2 / -0,036 |
| limpio 17386 | AFIB | 17,8 / 0,993 | 17,9 / 0,993 | -0,3 / -0,001 | -5,1 / -0,016 | -8,7 / -0,048 |
| limpio 17690 | CRBBB_CLBBB | 20,3 / 0,995 | 20,3 / 0,995 | -0,3 / +0,000 | -3,4 / -0,005 | -5,7 / -0,012 |
| limpio 2433 | CRBBB_CLBBB | 20,1 / 0,995 | 20,4 / 0,996 | -1,3 / -0,001 | -3,2 / -0,004 | -7,6 / -0,021 |
| limpio 6231 | IMI_AMI | 18,6 / 0,994 | 18,7 / 0,994 | -1,1 / -0,002 | -4,1 / -0,009 | -8,9 / -0,041 |
| limpio 18506 | IMI_AMI | 20,3 / 0,995 | 20,3 / 0,995 | -0,1 / -0,000 | -1,9 / -0,003 | -7,4 / -0,020 |
| limpio 19424 | SBRAD_STACH | 19,4 / 0,995 | 19,4 / 0,995 | -1,4 / -0,002 | -5,1 / -0,012 | -7,8 / -0,027 |
| limpio 8507 | SBRAD_STACH | 18,8 / 0,993 | 19,0 / 0,994 | +0,0 / +0,000 | -2,7 / -0,006 | -7,0 / -0,025 |
| distorsionado 1172 | NORM | 15,4 / 0,987 | 15,4 / 0,987 | -0,2 / +0,000 | -2,6 / -0,013 | -5,9 / -0,044 |
| distorsionado 1355 | NORM | 13,9 / 0,980 | 13,6 / 0,980 | -0,9 / -0,005 | -4,4 / -0,042 | -15,2 / -0,760, compuerta leads-missing-from-template |
| distorsionado 8645 | NORM | 14,2 / 0,983 | 14,6 / 0,984 | -1,0 / -0,005 | -3,6 / -0,022 | -7,3 / -0,080 |
| distorsionado 7221 | AFIB | 15,6 / 0,987 | 15,3 / 0,986 | -0,3 / -0,002 | -2,8 / -0,015 | -6,1 / -0,048 |
| distorsionado 8215 | AFIB | 13,9 / 0,981 | 13,7 / 0,979 | -0,7 / -0,003 | -3,4 / -0,027 | -13,8 / -0,582, sin compuerta |
| distorsionado 17386 | AFIB | 14,5 / 0,985 | 14,9 / 0,986 | -1,5 / -0,006, tira en V1 sin compuerta | -4,6 / -0,029 | -18,4 / -0,819, layout desconocido |
| distorsionado 17690 | CRBBB_CLBBB | 19,3 / 0,995 | 18,9 / 0,994 | -1,7 / -0,003 | -3,1 / -0,006 | -21,2 / -0,865, compuerta leads-missing-from-template |
| distorsionado 2433 | CRBBB_CLBBB | 17,1 / 0,990 | 16,9 / 0,990 | -0,5 / -0,001 | -2,9 / -0,010 | -6,9 / -0,040 |
| distorsionado 6231 | IMI_AMI | 14,6 / 0,986 | 14,8 / 0,986 | -1,1 / -0,004 | -4,2 / -0,023 | -17,0 / -0,947, sin compuerta |
| distorsionado 18506 | IMI_AMI | 17,5 / 0,991 | 17,6 / 0,992 | -1,2 / -0,003 | -2,9 / -0,008 | -6,9 / -0,033 |
| distorsionado 19424 | SBRAD_STACH | 16,8 / 0,990 | 17,1 / 0,991 | -1,0 / -0,003 | -3,6 / -0,012 | -6,7 / -0,033 |
| distorsionado 8507 | SBRAD_STACH | 16,6 / 0,989 | 16,6 / 0,989 | -0,7 / -0,002 | -3,0 / -0,011 | -7,0 / -0,042 |

### Decisión

Ninguna reducción cumple la tolerancia. Se mantiene `resample_size: 3000`, es decir, la
segmentación a tamaño nativo para las imágenes que llegan hoy.

- 2000 ahorra un 8 % (3,7 s por imagen) y ya pierde 0,52 dB en limpio y 0,92 dB en
  distorsionado, con r 0,003 peor en distorsionado. Además, en el registro distorsionado 17386
  la tira de ritmo se atribuyó a V1 en vez de II, con r de -0,51 en esa derivación, y ninguna
  compuerta lo detectó porque V1 es una derivación de tira convencional. Es un error silencioso.
- 1600 ahorra un 40 % pero pierde unos 3,5 dB en las dos variantes.
- 1200 rompe la digitalización de las distorsionadas (SNR 4,75 dB, tres registros degradados
  y dos más con r de 0,04 y 0,40 que ninguna compuerta marcó) y ni siquiera es más rápida: la
  búsqueda del tamaño de píxel se triplica.

Entre 2000 y el tamaño nativo (2200 a 2400 px) el ahorro posible es menor del 8 % y la
tendencia ya es de pérdida, así que no se midieron escalas intermedias.

El valor queda configurable: `ECG_DIGITIZER_RESAMPLE_SIZE` lo sobreescribe por despliegue,
y cualquier cambio debe volver a pasar por esta validación.

## 3. Memoria del Mac y validez de los tiempos

Tras el estudio de escalas, una primera tanda de hilos se descartó. Con 4 hilos, cuatro
imágenes tardaron de 107 a 190 s cada una, y subieron también etapas que no dependen de los
hilos: el import de torch pasó de 1,3 a 5,8 s, la identificación de layout de unos 12 a 47 s
y el guardado del PNG de 6 a 21 s. La causa no era torch sino la memoria. El swap tenía 6,9
de 8 GB ocupados, el compresor guardaba unos 17 GB de páginas y, con el digitalizador
parado, el sistema seguía leyendo unas 1.300 páginas por segundo del swap. Ningún proceso
visible pasaba de 800 MB: la memoria la retienen, comprimida o en swap, las aplicaciones
abiertas (OrbStack, simulador, navegador, editores). Un digitalizador con 5 a 6,4 GB de pico
sobre esa base lleva la máquina al thrashing.

Consecuencias para leer este documento:

- Las métricas de calidad del estudio de escalas no dependen del tiempo y son válidas.
- Los tiempos del estudio de escalas se tomaron seguidos, con la misma carga para las cuatro
  escalas; sirven para compararlas entre sí, no como cifra absoluta del servidor.
- Las comparaciones de modo y de hilos de la sección 4 alternan las dos variantes (A B A B)
  para que la deriva de la máquina caiga igual sobre ambas, y registran cuántas páginas se
  leyeron del swap durante cada corrida.
- La memoria pico de un análisis, de 5 a 6,4 GB, es en sí misma el límite práctico de este
  Mac. En el servidor de 16 GB, con el worker como carga principal, cabe con holgura; aquí no.

## 4. Optimizaciones, medidas por separado

Todas sobre la resolución elegida (3000, tamaño nativo). Durante estas corridas el sistema
leyó del swap entre 180.000 y 1.280.000 páginas por corrida, así que las diferencias
pequeñas están dentro del ruido y se reportan las dos rondas.

### a. Hilos de PyTorch

`ECG_TORCH_THREADS` fija los hilos de torch. En el digitalizador llega como
`OMP_NUM_THREADS` y `MKL_NUM_THREADS` en el entorno del proceso hijo, que torch lee al
arrancar; en ECGFounder se aplica con `torch.set_num_threads` antes de construir el modelo.
Por defecto: las CPU disponibles para el proceso (en Linux respeta el cpuset del contenedor),
con tope de 8.

Una imagen de 2391x1338, solo digitalización, con PNG de depuración:

| Hilos | Ronda 1 | Ronda 2 | Segmentación | Identificación de layout |
|---:|---:|---:|---:|---:|
| 8 | 37,4 s | 36,6 s | 10,8 / 10,6 s | 8,6 / 8,2 s |
| 4 | 42,3 s | 39,1 s | 12,6 / 12,4 s | 11,0 / 9,2 s |

Con 8 hilos, un 9 % más rápido que con 4, y la ganancia se concentra en las dos redes.
En este Mac (8 núcleos físicos, 16 lógicos) el valor por defecto coincide con el de torch,
así que aquí no cambia nada; lo que aporta es que el servidor use un número decidido y
ajustable. Las pruebas con 12 y 16 hilos se hicieron en la tanda descartada por el swap
(sección 3) y no se repitieron. En el servidor de 8 vCPU compartidas conviene medir 4 y 8 con
la carga real antes de fijarlo.

### b. Digitalizador en memoria

`ECG_DIGITIZER_MODE=persistent`, el nuevo valor por defecto, arranca una vez
`python -m src.serve` (parche 0004), que carga los modelos y atiende un estudio por petición
por stdin y stdout. Sigue siendo un proceso aparte, así que la frontera de licencia de
`digitizer.py` no cambia. Si el proceso muere, la llamada en curso falla con
`DigitizerFailed` y la siguiente arranca uno nuevo. `ECG_DIGITIZER_MODE=subprocess` conserva
el comportamiento anterior. La interfaz pública (`pipeline.run`, `digitizer.digitize`) no cambia.

Dos estudios seguidos con la imagen normal y la de fibrilación de ejemplo, cada uno con su
propia llamada a `pipeline.run` como hace el worker de api-EKG, digitalización más ECGFounder,
alternando los modos:

| Modo | Ronda | Estudio 1 | Estudio 2 | RSS pico |
|---|---:|---:|---:|---:|
| subprocess | 1 | 47,4 s | 44,8 s | 5,5 GB |
| persistent | 1 | 56,7 s | 41,2 s | 5,8 GB |
| subprocess | 2 | 54,0 s | 45,6 s | 5,6 GB |
| persistent | 2 | 47,6 s | 38,7 s | 6,0 GB |

El primer estudio del modo persistente paga la carga, igual que el subproceso. Del segundo en
adelante ahorra entre 3,6 y 6,9 s por estudio, unos 5 s de media (un 11 %), que coincide con
los 4,9 s de imports y carga de pesos del perfil. La memoria pico sube unos 300 MB, y el
proceso retiene los modelos entre estudios.

Los CSV de los dos modos son idénticos byte a byte en los cuatro casos, y
`scripts/integration_check.py` pasa en modo persistente, con correlación 1,0000 y la misma
cobertura en las doce derivaciones.

### c. Resolución de trabajo

`ECG_DIGITIZER_RESAMPLE_SIZE` sobreescribe `MODEL.KWARGS.resample_size` por despliegue; el
valor por defecto sigue en `configs/digitizer_cpu.yml` y queda en 3000 por lo medido en la
sección 2. No aporta velocidad con el valor elegido: deja la palanca documentada y validada.

## 5. Antes y después

Por estudio, imagen de ejemplo ampliada a 2391x1338, digitalización más ECGFounder, en este Mac:

| Configuración | Tiempo por estudio |
|---|---:|
| Medición previa, máquina cargada | unos 99 s |
| main, subproceso, máquina en reposo (sección 1) | 56,9 s |
| Rama, subproceso, 8 hilos (sección 4b) | 44,8 a 54,0 s |
| Rama, persistente, estudios después del primero | 38,7 a 41,2 s |

El análisis queda por debajo del minuto en este Mac cuando la memoria lo permite, sin
cambiar la calidad de la digitalización: las salidas son idénticas y la resolución no se tocó.

## 6. Decisión y lo que queda

- Reescalado: se mantiene 3000. Ninguna reducción cumple la tolerancia.
- Hilos: configurables, por defecto las CPU disponibles con tope de 8.
- Digitalizador persistente por defecto, con el subproceso como alternativa.

Siguiente ahorro medido pero no aplicado por defecto: el PNG de depuración que escribe el
digitalizador cuesta de 5,3 a 6 s por imagen, un 12 % del estudio, y no lo lee nadie: api-EKG
no lo usa y el pipeline solo lo borra. Pasar `DATA.save_mode` a `timeseries_only` no toca el
CSV ni los metadatos. Sigue activo por defecto porque alguien podría estar mirándolo a mano;
`ECG_DIGITIZER_DEBUG_PNG=0` lo desactiva por despliegue.

No medido:

- El servidor de destino. Los tiempos absolutos del Mac no se trasladan y los hilos deben
  medirse allí.
- Hilos 12 y 16, perdidos en la tanda con thrashing.
- El set completo de 48 imágenes: el estudio usó el subconjunto de 24 descrito en la sección 2.
- La validación completa del subconjunto con la configuración final. No hace falta para la
  calidad, porque los CSV salen idénticos en ambos modos y la resolución no cambió, pero no
  se corrió.

## 7. GPU

Medido en Colab con una GPU T4 con `scripts/medir_gpu_colab.ipynb` (PR #16), sobre las 24
imágenes del estudio de escalas (sección 2), con resolución 3000, modo persistente y sin PNG
de depuración. La CPU de referencia es la de la misma máquina de Colab, no la de este Mac.

| Configuración | Tiempo por estudio |
|---|---:|
| CPU de Colab | unos 95 s |
| GPU T4, en caliente | unos 17 s |
| GPU T4, primer estudio (en frío) | unos 26 s |

La calidad de la digitalización en GPU es idéntica a la de CPU en las 24 de 24 imágenes.

Cómo se activa: `ECG_DEVICE=cuda` y nada más. El pipeline genera una copia de
`configs/digitizer_cpu.yml` con `MODEL.KWARGS.device` y
`MODEL.KWARGS.config.LAYOUT_IDENTIFIER.KWARGS.device` en `cuda` (en el directorio temporal
del sistema, con un hash del contenido en el nombre) y se la pasa al digitalizador, en modo
persistente o subproceso; ECGFounder recibe el mismo dispositivo. Con `cpu` se usa el archivo
del repositorio sin tocarlo. Si se pide `cuda` y torch no ve la GPU, `pipeline.run` falla
antes de leer la primera imagen con `DeviceUnavailable`, y el worker de api-EKG al arrancar.

No medido todavía:

- El pipeline con esta integración en una GPU real: el notebook cambiaba la configuración a
  mano y usaba torch 2.3.1 con CUDA 12.1; la imagen GPU de api-EKG usa torch 2.2.2 con CUDA
  12.1, el mismo torch que la imagen CPU. Conviene correr `scripts/integration_check.py` con
  `ECG_DEVICE=cuda` en la primera máquina con GPU.
- La memoria de GPU con dos workers en la misma tarjeta.
