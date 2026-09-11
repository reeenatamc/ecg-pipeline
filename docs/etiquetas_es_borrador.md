# Borrador de traducción de etiquetas ECGFounder (español, Ecuador)

## Qué es este archivo

`docs/etiquetas_es_borrador.csv` es una primera propuesta de traducción al español de las
150 etiquetas diagnósticas que emite ECGFounder (`ecg_pipeline/interpret/tasks.txt`), listadas
en inglés como enunciados del formato GE Marquette 12SL. La aplicación muestra hoy esas
etiquetas tal como el modelo las produce, sin traducir.

Este es un borrador para revisión clínica, no una traducción final ni una decisión de
producto. Cada fila debe leerse, corregirse o descartarse por un cardiólogo antes de que
cualquiera de estos textos llegue a un usuario. El archivo no está conectado a la aplicación:
`contract.py` sigue entregando las etiquetas del modelo sin reescribir (ver README, sección
"Feeding app-EKG"), y así debe seguir hasta que exista una decisión clínica y de producto
sobre qué mostrar y cómo.

## Columnas del CSV

- `label_en`: la etiqueta en inglés, copiada de forma idéntica desde `tasks.txt`, en el mismo
  orden (150 filas, ni una más).
- `propuesta_es`: la traducción propuesta.
- `categoria`: una de ritmo, conduccion, eje, hipertrofia, isquemia_infarto, repolarizacion,
  marcapasos, tecnico, resumen, otro.
- `mostrar_al_usuario`: si, no o revisar. "No" se usa para enunciados de resumen o
  comparación puros (por ejemplo ABNORMAL ECG, NORMAL ECG) que no deberían mostrarse junto a
  hallazgos específicos. "Revisar" marca las filas donde no hay certeza, sobre todo
  fragmentos de plantilla que no tienen sentido por sí solos.
- `nota`: por qué se eligió ese término, la fuente en la que se apoya, o "sin equivalente
  directo" cuando corresponde.

## Fuentes usadas

- Glosario de términos y abreviaturas en cardiología (inglés-español) de la Sociedad Española
  de Cardiología.
- Revista Española de Cardiología: traducción de la guía ESC 2020/2024 de fibrilación
  auricular y de la guía ESC 2020 de síndrome coronario agudo sin elevación del ST
  (NSTE-ACS), además de artículos sobre bloqueo sinoauricular, trastorno inespecífico de la
  conducción intraventricular y efecto digitálico citados abajo.
- SIAC, Texto de Cardiología, para terminología general de arritmias y bloqueos.
- Búsquedas puntuales en la web para confirmar términos dudosos (referencias citadas en la
  columna `nota` cuando aplica).

## Una nota sobre los enunciados de plantilla

Alrededor de 59 de las 150 filas quedaron marcadas `revisar`. La mayoría no es una duda de
vocabulario sino una limitación de origen: ECGFounder usa como clases de salida los
enunciados del sistema GE Marquette 12SL, y muchos de esos enunciados son fragmentos de una
plantilla más larga, pensados para completarse con una lista de derivaciones ("... IN") o
para compararse con un ECG previo ("QT HAS LENGTHENED", "ST NOW DEPRESSED IN"), o para
adjuntarse a un enunciado de ritmo principal ("WITH RAPID VENTRICULAR RESPONSE", "WITH 1ST
DEGREE AV BLOCK"). Aislados, ninguno de esos fragmentos es una frase clínica completa ni en
inglés ni en español. La traducción literal se hizo de todos modos para que el cardiólogo
tenga un punto de partida, pero la decisión real (mostrarlos combinados con el hallazgo
principal, descartarlos, o reformular el enunciado del modelo en un hallazgo nuevo) es de
producto y de criterio clínico, no de traducción.

## Las etiquetas más difíciles

1. **RIGHTWARD AXIS / LEFTWARD AXIS** frente a **RIGHT AXIS DEVIATION / LEFT AXIS DEVIATION**.
   La plantilla 12SL distingue dos grados de desviación del eje, pero la bibliografía en
   español no tiene dos términos separados y establecidos para esa distinción. Se tradujeron
   como "eje desviado hacia la derecha/izquierda" para diferenciarlos de "desviación del eje
   a la derecha/izquierda", pero es una solución propia, no tomada de una fuente.

2. **WITH 2ND DEGREE SA BLOCK MOBITZ I / MOBITZ II** (bloqueo sinoauricular, no
   auriculoventricular). Se optó por "bloqueo sinoauricular de segundo grado tipo Mobitz
   I/II" porque la nomenclatura Mobitz se documenta también para el nodo sinoauricular, pero
   es un hallazgo mucho menos frecuente en la literatura en español que el bloqueo AV
   homónimo, y existe riesgo real de que un lector lo confunda con este último.

3. **PULMONARY DISEASE PATTERN**. Se tradujo como "patrón de enfermedad pulmonar", pero en
   electrocardiografía este hallazgo también se describe como "patrón de cor pulmonale" o
   "patrón de EPOC" según la fuente, sin un término único consolidado.

4. **SUSPECT UNSPECIFIED PACEMAKER FAILURE**. Además de la duda entre "falla" (uso habitual
   en Ecuador y Latinoamérica) y "fallo" (España), es una etiqueta de alta prioridad clínica
   con un calificador deliberadamente vago ("unspecified") que quizá no convenga suavizar en
   la traducción.

5. **RSR' OR QR PATTERN IN V1 SUGGESTS RIGHT VENTRICULAR CONDUCTION DELAY**. Es el enunciado
   más largo y más técnico de las 150 etiquetas. Se tradujo de forma cercana al original
   ("patrón RSR' o QR en V1, sugestivo de retraso de la conducción ventricular derecha"), pero
   convendría que el cardiólogo decida si prefiere una forma más breve para mostrar en la app.

6. **R IN AVL / BLOCKED / ACUTE / ANTEROLATERAL LEADS**. Cuatro fragmentos huérfanos de la
   plantilla 12SL (parte de un criterio de voltaje, de una extrasístole bloqueada, de un
   calificador de agudeza y de una lista de derivaciones, respectivamente) que no son frases
   completas en ningún idioma. Se tradujeron literalmente, marcados "revisar", pero
   probablemente no deberían mostrarse solos bajo ninguna traducción.

7. **LEFT ANTERIOR FASCICULAR BLOCK / LEFT POSTERIOR FASCICULAR BLOCK**. El glosario de la SEC
   prefiere "bloqueo fascicular anterior/posterior izquierdo", pero en la práctica clínica en
   español (y en Ecuador en particular) es igual de común o más común "hemibloqueo
   anterior/posterior izquierdo". Se dejó la forma del glosario con una nota mencionando la
   alternativa, para que el cardiólogo elija.

8. **SINUS/ATRIAL CAPTURE**. Se tradujo como "captura sinusal/auricular", un término correcto
   de electrofisiología (un latido capturado durante una disociación AV u otro ritmo
   competitivo), pero poco transparente para cualquier lector que no sea especialista, lo que
   plantea la pregunta de si conviene mostrarlo tal cual o reformularlo.

9. **OR DIGITALIS EFFECT**. Es el fragmento de un enunciado disyuntivo del 12SL (algo como
   "cambios de ST-T compatibles con isquemia... OR DIGITALIS EFFECT", es decir, un
   diagnóstico diferencial, no una conjunción). "Efecto digitálico" es un término estándar y
   bien documentado, pero el fragmento aislado no comunica que se trata de una alternativa
   diagnóstica y no de un hallazgo adicional.

10. **ACUTE MI / STEMI** frente a **ACUTE MI**. Son dos etiquetas separadas en `tasks.txt`;
    la primera se tradujo como IAMCEST (siguiendo el término dado en el enunciado de la
    tarea) y la segunda como "infarto agudo de miocardio" sin especificar. Queda a criterio
    del cardiólogo si conviene que ambas etiquetas se muestren de forma distinguible en la
    app o si una de las dos debería fusionarse con la otra.
