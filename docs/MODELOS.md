# Los dos modelos: instalación, formatos y cómo se conectan

Documento de traspaso. Responde a cuatro preguntas concretas: cómo se instaló cada modelo,
qué formato pide ECGFounder, qué hizo falta para conectar la salida del digitalizador con
su entrada, y cómo se configuró el modelo.

Todos los números que aparecen aquí están medidos sobre este repositorio y una corrida
real del 2026-08-06, no estimados.

---

## 0. El mapa, en diez líneas

Son dos modelos independientes, de dos equipos distintos, con dos licencias distintas y
dos historias de instalación completamente distintas:

| | **Open-ECG-Digitizer** | **ECGFounder** |
|---|---|---|
| Qué hace | foto de un ECG en papel → serie temporal | serie temporal → 150 clases diagnósticas |
| Quién | Ahus-AIM (Noruega) | PKUDigitalHealth (Pekín) |
| Licencia | **CC BY-SA 4.0** (copyleft) | **MIT** (permisiva) |
| Cómo se usa aquí | **subproceso externo**, nunca se importa | **importado**, `net1d.py` copiado literal |
| Pesos | en el repo, vía git-lfs | descargados de HuggingFace (~370 MB ×2) |

Esa diferencia de licencia es la que explica por qué uno se clona aparte y el otro se
copia dentro. No es una preferencia de estilo: CC BY-SA es contagiosa y MIT no.

---

## 1. ¿Cómo se instalaron?

### Open-ECG-Digitizer — clonado aparte, no incluido

```bash
bash scripts/setup_digitizer.sh
```

Ese script hace tres cosas: clona `https://github.com/Ahus-AIM/Open-ECG-Digitizer.git`
como carpeta hermana, ejecuta `git lfs pull` (los pesos del U-Net vienen por LFS; sin
git-lfs te bajas punteros de texto en vez de modelos), y aplica un parche de portabilidad.

**No está copiado dentro de este repositorio, y es deliberado.** CC BY-SA 4.0 obliga a que
cualquier "Adapted Material" se publique bajo la misma licencia. Copiar su código aquí
convertiría a este repo en obra derivada y arrastraría el copyleft a todo lo demás
—incluido el backend y la app—. Se ejecuta como programa separado desde
`ecg_pipeline/digitizer.py`, que localiza el checkout por `$OPEN_ECG_DIGITIZER_HOME` o como
directorio hermano.

Dos consecuencias que conviene decir en voz alta:

- Ejecutar un programa no es adaptarlo, y CC BY-SA no tiene cláusula de red (a diferencia
  de AGPL). Servirlo detrás de un backend **no** dispara ShareAlike.
- **La salida no está contaminada.** La serie temporal digitalizada es dato producido por
  ejecutar el software, no obra derivada.

### ECGFounder — parcialmente incluido

```bash
bash scripts/download_weights.sh
```

Descarga los dos `.pth` de HuggingFace a `weights/`. Son ~370 MB cada uno y están en
`.gitignore`: binarios de ese tamaño no van en un repo.

De ECGFounder sí hay código dentro: `ecg_pipeline/interpret/net1d.py` es **copia literal**
de su definición de red, y `tasks.txt` es su lista de 150 clases. MIT lo permite siempre
que se conserve el aviso de copyright, y se conserva (ver `NOTICE`).

### Problemas de compatibilidad — sí, cuatro, y todos reales

**1. numpy y torch están clavados juntos, y no por precaución.**

```
numpy>=1.26,<2
torch>=2.2,<2.4
```

torch 2.2.x está compilado contra la ABI de C de numpy 1.x y **falla al importar** bajo
numpy 2.x. Relajas una cota y tienes que relajar la otra: numpy≥2 exige torch≥2.3. No es
una recomendación, es una incompatibilidad registrada.

**2. torchvision 0.17.2 es la última build para Intel macOS.** Su
`torchvision.io.decode_image` no acepta ni una ruta de archivo ni un `mode` como cadena,
que es justo como lo llama el digitalizador. Parcheado para usar PIL
(`patches/0001-digitizer-portability.patch`).

**3. El digitalizador importa `ray.tune` al cargar `src/utils.py`.** `ray` es dependencia
solo de entrenamiento, y exigirla en tiempo de import rompe una instalación de solo
inferencia. El mismo parche la hace opcional.

**4. El Python de pyenv está compilado sin liblzma**, así que el venv lleva un `_lzma.py`
de relleno en site-packages o torchvision no importa.

> Los parches 2 y 3 son **arreglos genéricos, no específicos de este proyecto**. Lo ideal
> sería mandarlos como PR a Ahus-AIM y que la carpeta `patches/` desaparezca. Ojo: un
> parche contra CC BY-SA es modificación de material ShareAlike, así que el contenido de
> `patches/` está bajo CC BY-SA, no bajo la licencia de este repo.

### El intérprete

Un solo venv, el que vive dentro del checkout del digitalizador
(`Open-ECG-Digitizer/venv`), Python 3.12, CPU, sin GPU. Instala ahí con `--no-deps` o con
una restricción: cualquier paquete que declare `torch` o `numpy` a secas puede mover el par
clavado sin avisar.

---

## 2. ¿Qué formato necesita ECGFounder?

**No un CSV. Un array de numpy.** El CSV es la salida del digitalizador; convertirlo es
trabajo nuestro.

La firma real (`interpret_ecg.py`):

```python
@torch.no_grad()
def predict_probs(model, signal, device="cpu"):
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]                        # (L,) -> (1, L)
    x = torch.from_numpy(arr).unsqueeze(0).to(device)   # -> (1, C, L)
    return torch.sigmoid(model(x)).cpu().numpy().ravel()
```

Es decir:

| | |
|---|---|
| **Tipo** | `np.float32` |
| **Forma** | `(C, L)` — canales × muestras. Acepta también `(L,)` para una derivación |
| **C** | `1` con el checkpoint de 1 derivación, `12` con el de 12. No hay otra opción |
| **L** | **variable** |
| **Normalización** | z-score global: `(x - mean) / (std + 1e-8)` |
| **Unidades** | irrelevantes — el z-score las cancela |
| **Salida** | 150 probabilidades sigmoide, alineadas por índice con `tasks.txt` |

**Que `L` sea variable no es un detalle menor.** `Net1D` termina en *global average
pooling*, así que traga cualquier longitud. Eso permitió lo importante: **no estirar** una
derivación de 2,5 s hasta 10 s. Estirarla habría multiplicado por cuatro la frecuencia
cardiaca aparente. Cada derivación conserva su base de tiempo nativa.

---

## 3. ¿Hubo problemas al conectar las dos salidas?

Sí, pero **no fueron de formato**. El formato es trivial. El problema fue semántico y es el
que se llevó el diseño entero.

### Qué escribe el digitalizador

Un `<nombre>_timeseries_canonical.csv`. Medido sobre un archivo real de la corrida:

```
I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6
nan,nan,nan,nan,nan,nan,118.479309,nan,288.213470,222.376510,95.274452,nan
nan,nan,nan,nan,nan,nan,119.167374,nan,286.149902,223.684753,96.574738,nan
```

- **Filas = tiempo, columnas = derivación** (traspuesto respecto a lo que quiere el modelo)
- 5000 filas de datos = **10,0 s a 500 Hz**
- Valores en **microvoltios** (el contrato de la app los quiere en milivoltios: ÷1000)
- **`nan` donde esa derivación no se imprimió**

### El problema de verdad: la mitad de la matriz son NaN

En ese archivo: **30 012 celdas NaN de 60 000**. El 50%.

No es un fallo. Es la física del papel: en una impresión 3×4 estándar cada derivación de la
rejilla solo ocupa 2,5 de los 10 segundos. Medido sobre nuestra corrida:

```
   I : 0,00–2,48 s
 III : 0,00–2,48 s
 aVR : 2,51–4,98 s
  II : 0,00–8,31 s  +  9,76–10,00 s     <- tira de ritmo, completa
```

Un modelo al que le das 10 s con 7,5 s de NaN no devuelve nada útil. Y **rellenar los
huecos con ceros o interpolando entre tramos es mentir sobre el paciente**: dibuja una
línea perfectamente creíble sobre un intervalo donde nadie sabe qué hacía el corazón.

### Qué transformaciones hicieron falta

En `ecg_pipeline/interpret/waveform.py`:

1. **Traspuesta** — `load_canonical_csv` devuelve `(n_derivaciones, n_muestras)`.
2. **`clean_lead`** — recorta al tramo válido `[primera..última]` e interpola los NaN
   *internos*. **No reescala ni estira**: conserva la base de tiempo, y con ella la
   frecuencia cardiaca.
3. **`zscore`** — media y desviación globales. Sobre 12 derivaciones se hace global a
   propósito, para preservar las amplitudes relativas entre derivaciones, que son
   diagnósticamente significativas.

Nada más. Sin conversión de unidades (el z-score las cancela), sin remuestreo.

### Y la consecuencia de diseño: cuatro caminos, no uno

Como no se puede inventar señal, hay cuatro `--pathway`:

| pathway | qué hace | para qué sirve |
|---|---|---|
| **`rhythm`** (por defecto) | checkpoint de 1 derivación sobre cada tira de ritmo completa (II/V1/V5), promediando opiniones | **ritmo y frecuencia**. Es el fiable |
| `1lead` | 1 derivación elegida a mano | inspección |
| `morphology` | latido representativo (mediana) por derivación, alineado en fase entre las doce, repetido hasta 10 s → checkpoint de 12 | **morfología**. **Borra el ritmo** (un latido mediano es perfectamente regular) |
| `12lead` | montaje ingenuo de las ventanas de cada columna | **comparación histórica, no usar** |

**`12lead` sobrellama patología** y por eso se marca degradado incondicionalmente: en un 3×4
las columnas se registran en momentos distintos, así que el montaje está desalineado en
fase. Sobre un ECG verificado como normal devolvía `LATERAL INFARCT 0.997`.

---

## 4. ¿Cómo se configuró el modelo?

**Pesos preentrenados directamente. Cero entrenamiento, cero fine-tuning.**

### Los constructores tienen que coincidir exactamente

```python
_COMMON_KWARGS = dict(
    base_filters=64, ratio=1,
    filter_list=[64, 160, 160, 400, 400, 1024, 1024],
    m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
    kernel_size=16, stride=2, groups_width=16,
    use_bn=False, use_do=False, n_classes=150,
)
MODEL_KWARGS_1LEAD  = dict(in_channels=1,  **_COMMON_KWARGS)
MODEL_KWARGS_12LEAD = dict(in_channels=12, **_COMMON_KWARGS)
```

Copiados de `ptbxl_eval.py` / `finetune_model.py` de upstream. **Lo único que cambia entre
los dos checkpoints es `in_channels`.** Si te desvías de estos valores el `state_dict` no
encaja, y como se carga con `strict=False` **no te va a explotar**: te va a cargar mal y a
devolver números plausibles. Por eso se imprime un aviso a stderr contando claves que
faltan o sobran. Si ves ese aviso, algo está mal.

### Las probabilidades NO están calibradas

Esto es lo más fácil de malinterpretar del sistema entero. Las salidas son sigmoides sin
calibrar: `ABNORMAL ECG` puntúa alto **a la vez** que `NORMAL SINUS RHYTHM`. En nuestra
corrida sobre un ECG normal:

```
0.989  SINUS RHYTHM
0.957  ABNORMAL ECG          <- a la vez que
0.951  NORMAL SINUS RHYTHM
```

No es una contradicción del modelo, es que **0.5 no es el umbral**. El propio
`ptbxl_eval.py` de ECGFounder binariza con *umbrales óptimos por clase* derivados sobre
PTB-XL. Sin umbrales, lo que sale es **un ranking, no un veredicto** — y así se trata en
todo el código.

Se pasan con `--thresholds umbrales.json` (un `{etiqueta: umbral}`) o un `--threshold` plano.

---

## 5. Tres cosas abiertas que deberías saber

**1. El filtrado de ECGFounder no está implementado.** El `NOTICE` afirma que
«upstream requiere que se siga el preprocesado de su `dataset.py` (filtrado, normalización
z-score); `interpret_ecg.py` lo implementa». **Solo está el z-score.** Busqué
`butter|bandpass|filtfilt|sosfilt|notch` en todo el módulo de interpretación: no hay
ninguno.

Puede que dé igual —una señal digitalizada de papel ya viene limitada en banda por la
impresión y el escaneo, y no arrastra red eléctrica ni deriva de línea base como un
registro crudo— pero **nadie lo ha medido**. O se implementa el filtro, o se corrige el
NOTICE. Ahora mismo el documento promete más de lo que el código hace.

**2. No hay datos de validación.** Nada de esto está validado contra ECG con verdad de
referencia. Los números de arriba son de ejemplos sueltos. Es el bloqueo real del proyecto,
por encima de cualquier detalle de ingeniería.

**3. Citación obligatoria.** El README del digitalizador la exige para uso en investigación:

```bibtex
@article{stenhede_digitizing_2026,
  title   = {Digitizing electrocardiograms from images},
  journal = {npj Digital Medicine},
  year    = {2026},
  doi     = {10.1038/s41746-025-02327-1}
}
```

---

## Para arrancar de cero

```bash
git clone <este-repo> && cd ecg-pipeline
bash scripts/setup_digitizer.sh      # clona + git-lfs + parche
bash scripts/download_weights.sh     # 2 × 370 MB de HuggingFace
pip install -r requirements.txt      # respeta las cotas, ver §1

python -m ecg_pipeline --images entrada/ --out salida/
```

Lecturas siguientes: `NOTICE` para el detalle de licencias, `patches/README.md` para qué
hace cada parche y por qué, y el README para las puertas de calidad y el flag
`--fail-on-degraded`.
