# Lab Semana 4 — Seq2Seq Inglés→Español desde cero

Modelo Encoder–Decoder LSTM implementado únicamente con tensores de PyTorch: sin
`nn.LSTM`, sin `nn.Embedding`, sin `nn.Linear` y **con backward manual** (nada de
`loss.backward()`).

```
oración_EN -> [E_enc] -> [Encoder LSTM] -> c = h_T^enc
                                            |
<SOS>+oración_ES -> [E_dec] -> [Decoder LSTM] -> [W_out+softmax] -> predicción
```

Curso: Deep Learning (UVG). Entregable: `S4 - Lab_Semana4_Seq2Seq_Estudiante.ipynb`
ejecutado, con todas las salidas visibles.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate en Linux/macOS
pip install -r requirements.txt
```

Ejecutar el notebook completo de arriba a abajo (las celdas dependen del estado
de las anteriores):

```bash
jupyter notebook "S4 - Lab_Semana4_Seq2Seq_Estudiante.ipynb"
```

o sin abrir la interfaz:

```bash
jupyter nbconvert --to notebook --execute --inplace "S4 - Lab_Semana4_Seq2Seq_Estudiante.ipynb"
```

Tarda ~1 minuto en CPU.

## Contenido

| Bloque | Qué hace |
|--------|----------|
| 0  | Corpus de 78 pares, split 75/25, vocabularios y pesos iniciales (dado) |
| 1  | `get_embedding` — lookup de columna de la matriz de embeddings |
| 2  | `lstm_cell` (5 ecuaciones) y `encoder_forward` |
| 3  | `decoder_forward` con teacher forcing |
| 4  | `compute_loss` — entropía cruzada categórica promedio |
| 5  | Backward del decoder (BPTT sobre S pasos) |
| 6  | Backward del encoder (BPTT sobre T pasos) |
| 7  | Gradiente de embeddings: matriz completa vs diccionario disperso |
| 8  | Actualización SGD de todos los parámetros |
| 9  | Loop de entrenamiento: 5 iteraciones sobre `TRAIN_DATA` |
| 10 | `greedy_decode` — inferencia autoregresiva por argmax |
| 11 | Curva de pérdida → `convergencia_seq2seq.png` |
| 12 | Preguntas de análisis (1, 2 y 3) |
| 13 | Nota automática de la sección de código |

## Resultados

Verificación automática: **60/60 puntos** de código (bloques 2, 4, 5, 6, 7, 8 y 9
en CORRECTO).

Convergencia sobre 5 iteraciones:

| Iteración | Loss promedio |
|-----------|---------------|
| 1 | 5.0730 |
| 2 | 5.0301 |
| 3 | 4.9875 |
| 4 | 4.9452 |
| 5 | 4.9032 |

Reducción de 3.35%, monótona decreciente.

Precisión de primera palabra en test: **0/20 (0.0%)**. Es el resultado esperado
con los hiperparámetros que fija el enunciado: 5 iteraciones a `alpha = 0.01`
sobre 58 pares son ~290 actualizaciones, muy lejos de lo necesario para que el
modelo aprenda a traducir. La pérdida apenas baja de 5.07 a 4.90 (ln(163) ≈ 5.09,
que es la pérdida de una distribución uniforme), así que el decoder todavía emite
`<EOS>` en el primer paso y las traducciones salen vacías. El bloque 10 es
informativo y no puntúa; para ver traducciones reales harían falta cientos de
épocas o un learning rate mayor, pero eso rompería la verificación del bloque 9.

## Por qué importa la asimetría de largo

El corpus está construido a propósito con tres tipos de par:

- **Español más corto** (57 de 78) — elisión del sujeto: *she sings well* → *canta bien*
- **Español más largo** (13) — perífrasis verbal: *the cat sleeps* → *el gato está durmiendo*
- **Mismo largo** (8)

El encoder corre `len(src)` pasos y el decoder `len(tgt)-1`, sin ninguna
suposición de que sean iguales. Esa independencia entre los dos largos es
exactamente la razón de ser de la arquitectura Encoder–Decoder.
