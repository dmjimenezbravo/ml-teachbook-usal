# 2.2 Clasificación

## Introducción

En clasificación, nuestro objetivo es predecir **categorías** (clases) en lugar de valores continuos. Ejemplos incluyen:
- Spam vs No-spam
- Enfermo vs Sano
- Gato vs Perro
- Positivo vs Negativo

## Regresión Logística

### De Regresión a Clasificación

Regresión lineal predice valores continuos sin restricción. Para clasificación, necesitamos:
- Salidas entre 0 y 1 (probabilidades)
- Decisiones claras (0 o 1)

Usamos la **función sigmoide** (o logística):

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

```
Función Sigmoide:

Probabilidad
    |
  1 |               ___----
    |           ---/
 0.5|        --/
    |      -/
  0 |____/
    +---+---+---+---> z
   -4  -2   0   2
```

**Propiedades clave**:
- Para $z \to -\infty$: $\sigma(z) \to 0$
- Para $z \to +\infty$: $\sigma(z) \to 1$
- Para $z = 0$: $\sigma(z) = 0.5$

### La Ecuación

Combinamos regresión lineal con la sigmoide:

$$\hat{p} = \sigma(\beta_0 + \beta_1 x_1 + \ldots + \beta_p x_p)$$

Donde $\hat{p}$ es la probabilidad predicha de la clase positiva (1).

### Función de Pérdida: Log Loss (Cross-Entropy)

No podemos usar MSE para clasificación (no es apropiado). En su lugar, usamos:

$$J(\beta) = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i)]$$

Donde:
- Si $y_i = 1$ (clase positiva real): penalizamos si $\hat{p}_i$ es bajo
- Si $y_i = 0$ (clase negativa real): penalizamos si $\hat{p}_i$ es alto

### Decisión

Una vez que tenemos probabilidades, decidimos:
- Si $\hat{p} > 0.5$: predecir clase 1
- Si $\hat{p} \leq 0.5$: predecir clase 0

## Matriz de Confusión

### Visualizar el Desempeño

```
                    Predicción
                 Positivo | Negativo
Actual  Positivo |   TP   |   FN
        Negativo |   FP   |   TN
```

Donde:
- **TP (Verdadero Positivo)**: Correctamente predicho como 1
- **TN (Verdadero Negativo)**: Correctamente predicho como 0
- **FP (Falso Positivo)**: Incorrectamente predicho como 1 (falsa alarma)
- **FN (Falso Negativo)**: Incorrectamente predicho como 0 (falta)

### Ejemplo Práctico

Detectar enfermedades:

```
                    Test
                 Positivo | Negativo
Realidad Positivo |  85   |   15     (85 personas enfermas correctamente
         Negativo |  10   |  890     identificadas, 15 no detectadas)
                           (10 falsos positivos, 890 verdaderos negativos)
```

## Métricas de Evaluación

### Precisión (Precision)

$$\text{Precisión} = \frac{TP}{TP + FP}$$

De todos los casos que predijimos positivos, ¿cuántos realmente lo eran?

**Uso**: Importante cuando los falsos positivos son costosos (ej: spam filtering - no queremos bloquear emails legítimos)

### Recall (Sensibilidad)

$$\text{Recall} = \frac{TP}{TP + FN}$$

De todos los casos positivos reales, ¿cuántos encontramos?

**Uso**: Importante cuando los falsos negativos son costosos (ej: detección de enfermedad - queremos detectar a todos los enfermos)

### Precisión vs Recall: Trade-off

Aumentar un generalmente disminuye el otro:

```
Umbral bajo (predecimos 1 más a menudo):
- Recall: alto (encontramos muchos positivos)
- Precisión: baja (muchos son incorrectos)

Umbral alto (predecimos 1 pocas veces):
- Recall: bajo (nos perdemos muchos positivos)
- Precisión: alta (cuando decimos 1, es correcto)
```

### F1-Score

Combina ambas en una sola métrica:

$$F1 = 2 \times \frac{\text{Precisión} \times \text{Recall}}{\text{Precisión} + \text{Recall}}$$

Es la **media armónica** de precisión y recall.

### AUC-ROC

**Curva ROC (Receiver Operating Characteristic)**:
- Grafica Recall (eje y) vs Tasa de Falsos Positivos (eje x)
- A diferentes umbrales de decisión

**AUC (Area Under Curve)**:
- Área bajo la curva ROC
- Rango: 0 a 1
- 1.0 = perfecto
- 0.5 = aleatorio
- < 0.5 = peor que aleatorio

## Árboles de Decisión

### Intuición

Un árbol de decisión toma decisiones recursivamente basado en características:

```
¿Horas estudiadas > 3?
     |
  Si |  No
  |  |  |
  |  |  ├─ Calificación = baja
  |  |
  |  └─ ¿Dificultad del examen?
        |
     Fácil | Difícil
        |   |
        |   └─ Calificación = media
        └─ Calificación = alta
```

### Ventajas

- **Interpretable**: fácil de explicar
- **No paramétrico**: no asume forma de datos
- **Maneja no linealidad**: naturalmente
- **Maneja características mixtas**: numéricas y categóricas

### Desventajas

- **Propenso a overfitting**: puede crecer demasiado profundo
- **Inestable**: pequeños cambios en datos pueden cambiar completamente el árbol

## Random Forest

### Idea: Ensemble de Árboles

En lugar de un árbol, creamos muchos árboles (por ejemplo, 100) con variación:
- Cada árbol se entrena en una muestra bootstrap (muestreo con reemplazo)
- Cada división usa un subconjunto aleatorio de características

### Predicción

```
Árbol 1 ──┐
Árbol 2 ──┼─→ VOTACIÓN → Predicción Final
  ...    ──┤
Árbol n ──┘
```

Para clasificación: clase que más árboles predicen
Para regresión: promedio de predicciones

### Por Qué Funciona

- **Reduce varianza**: promediar muchos árboles ruidosos
- **Mantiene bajo sesgo**: cada árbol puede ser complejo
- **Captura interacciones**: los árboles las aprenden naturalmente

## Support Vector Machines (SVM)

### Objetivo

Encontrar el **hiperplano** (línea en 2D, plano en 3D, etc.) que mejor separa las clases:

```
Clase A (o)          Clase B (x)
    o                    x
      o o              x x
  o       o ────────────── ← hiperplano óptimo
      o           x x
    o           x
              x
```

El "margen" es la distancia entre la línea y los puntos más cercanos.

### SVM intenta maximizar este margen.

### El Kernel Trick

Para datos no linealmente separables, SVM usa kernels para proyectar a dimensiones superiores:

```
Espacio 2D (no separable):    Espacio 3D (separable):
                              
    o x                           o
    x o                         x   o
    o x          kernel         o x
    x o      ──────────→
    o x                        x o o
    x o                      x o
    o x              x o
```

Kernels comunes:
- **Linear**: para datos separables linealmente
- **RBF (Radial Basis Function)**: para datos complejos
- **Polynomial**: para patrones polinomiales

## Gradient Boosting (XGBoost, LightGBM)

### Idea: Aprender de los Errores

A diferencia de Random Forest que entrena árboles en paralelo, Boosting los entrena **secuencialmente**:

1. Entrenar árbol 1 en datos originales
2. Entrenar árbol 2 enfocándose en errores del árbol 1
3. Entrenar árbol 3 enfocándose en errores combinados
4. ...

```
Árbol 1: predice algo, comete errores
           ↓
Árbol 2: aprende de esos errores
           ↓
Árbol 3: aprende de errores residuales
           ↓
Predicción = suma ponderada de todos
```

### Ventajas

- **Muy precisos**: estado del arte en competencias de ML
- **Maneja datos complejos**: captura patrones intrincados
- **Feature importance**: identifica características importantes

### Desventajas

- **Lento para entrenar**: secuencial, no paralelo
- **Muchos hiperparámetros**: requiere ajuste cuidadoso
- **Menos interpretable**: más difícil de entender

## Redes Neuronales: Introducción

### El Perceptrón

Una neurona artificial que intenta replicar una neurona biológica:

```
Entrada → Sumatorio → Activación → Salida
  x1 ──────┐
           │ suma ponderada    función
  x2 ──────┼──→ z = Σ w*x + b ──→ σ(z) ──→ salida
           │
  x3 ──────┘

Donde:
- w: pesos (lo que aprende)
- b: sesgo (bias)
- σ: función de activación (sigmoid, ReLU, etc.)
```

### De una Neurona a Redes

Combinamos múltiples neuronas en capas:

```
Entrada        Capas Ocultas        Salida
  x1   ────────────┐
       │           ├─ neurona ─┐
  x2   │────────────┤           ├─ neurona ─→ ŷ
       │           ├─ neurona ─┘
  x3   ────────────┘
```

Cada neurona aprende a detectar patrones, capas posteriores combinan estos patrones.

### Funciones de Activación

#### Sigmoid
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- Rango: [0, 1]
- Antiguo, menos usado ahora

#### ReLU (Rectified Linear Unit)
$$\text{ReLU}(z) = \max(0, z)$$
- Rápido de computar
- Más usado actualmente

#### Softmax (para múltiples clases)
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$
- Convierte scores en probabilidades
- Las probabilidades suman 1

### Entrenamiento: Backpropagation

El algoritmo de entrenamiento de redes neuronales:

1. **Forward Pass**: calcular salida
2. **Calcular pérdida**: qué tan lejos estamos del objetivo
3. **Backward Pass**: propagar el error hacia atrás
4. **Actualizar pesos**: con el error propagado

```
Entrada ──[forward]──→ Salida ──→ comparar con y_real
                          ↓
                    Calcular error
                          ↓
                      [backward]
                          ↓
                    Actualizar pesos
```

## Comparación de Algoritmos

| Algoritmo | Interpretable | Rápido | Preciso | Overfitting |
|-----------|---------------|--------|---------|------------|
| **Logística** | ✓✓ | ✓✓ | ○ | ○ |
| **Árbol** | ✓✓ | ✓✓ | ○ | ✗✗ |
| **SVM** | ✗ | ✗ | ✓✓ | ✓ |
| **Random Forest** | ○ | ✓ | ✓✓ | ✓ |
| **XGBoost** | ✗ | ✗ | ✓✓✓ | ✓ |
| **Red Neuronal** | ✗✗ | ✗ | ✓✓✓ | ✗ |

Donde ✓ = bueno, ✗ = malo, ○ = medio

## Pauta para Elegir Algoritmo

1. **Comienza simple**: Regresión Logística
2. **Si necesitas interpretabilidad**: Árbol de Decisión
3. **Si necesitas precisión**: Random Forest o XGBoost
4. **Si tienes datos muy complejos**: Red Neuronal
5. **Siempre**: valida con validación cruzada

## Resumen

- **Regresión Logística**: extensión de regresión lineal para clasificación
- **Matriz de Confusión**: visualiza el desempeño
- **Métricas**: elige según lo que importa (precisión o recall)
- **Árboles de Decisión**: interpretables pero proclives a overfitting
- **Random Forest**: ensemble que reduce overfitting
- **SVM**: encuentra separación óptima con kernel trick
- **Gradient Boosting**: estado del arte, requiere cuidado
- **Redes Neuronales**: poderosas pero complejas

---

**Siguiente**: En la Sección 3 exploraremos aprendizaje no supervisado, donde trabajamos con datos sin etiquetas.
