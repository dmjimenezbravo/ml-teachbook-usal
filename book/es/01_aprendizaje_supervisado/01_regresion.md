# 2.1 Regresión

## Introducción

La regresión es el problema de predecir un valor continuo. Empezamos con regresión lineal, el modelo más simple y fundamental del aprendizaje automático.

## Regresión Lineal: Lo Básico

### Intuición

Imagina que tienes datos sobre horas estudiadas y calificaciones de examen:

```
Horas Estudiadas | Calificación
        1        |      3
        2        |      5
        3        |      7
        4        |      8
        5        |      10
```

La regresión lineal intenta encontrar una línea recta que se ajuste mejor a estos puntos:

```
Calificación
    |
 10 |         ___o
    |       _/
  8 |     o/
    |    /
  7 |   o
    |  /
  5 |_o________
    |/  o
  3 |o
    |
    +--+--+--+--+--+---> Horas
    0  1  2  3  4  5
```

**La ecuación de la línea es:**
$$\hat{y} = \beta_0 + \beta_1 x$$

Donde:
- $\hat{y}$ es la predicción
- $\beta_0$ es el intercepto (dónde cruza el eje y)
- $\beta_1$ es la pendiente (qué tan rápido sube)
- $x$ es la entrada

### Generalización a Múltiples Características

Si tenemos múltiples características:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p$$

Por ejemplo, prediciendo precio de casa:
$$\text{precio} = \beta_0 + \beta_1 \times \text{metros}^2 + \beta_2 \times \text{habitaciones}$$

## Mínimos Cuadrados Ordinarios (OLS)

### ¿Cómo encontramos los mejores $\beta_0$ y $\beta_1$?

La idea es minimizar la suma de los cuadrados de los **residuos** (errores):

```
Dato real (y)
    |
    o (dato)
    |  |
    |  ±-- residuo = y - ŷ
    |  |
    ----ŷ (predicción)
```

**Objetivo:**
$$\min \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Esta es la función de pérdida que queremos minimizar.

### Solución Analítica

Para regresión lineal, existe una solución matemática exacta. En forma vectorial:

$$\beta = (X^T X)^{-1} X^T y$$

Donde:
- $X$ es la matriz de características
- $y$ es el vector de etiquetas
- $\beta$ es el vector de coeficientes

### Implementación Conceptual

```python
import numpy as np

# Datos
X = np.array([[1], [2], [3], [4], [5]])  # Horas
y = np.array([3, 5, 7, 8, 10])          # Calificaciones

# Agregar columna de 1s para el intercepto
X = np.column_stack([np.ones(len(X)), X])

# Calcular beta = (X^T X)^-1 X^T y
beta = np.linalg.inv(X.T @ X) @ X.T @ y

print(f"Intercepto: {beta[0]}")  # beta_0
print(f"Pendiente: {beta[1]}")   # beta_1

# Hacer predicción para 6 horas
X_new = np.array([1, 6])
prediccion = X_new @ beta
print(f"Predicción para 6 horas: {prediccion}")
```

## Descenso de Gradiente

### Motivación

La solución OLS funciona bien para regresión lineal, pero tiene un problema: requiere invertir una matriz, lo que es computacionalmente costoso cuando tenemos millones de características.

Existe un método más general: **Descenso de Gradiente**, que funciona para cualquier tipo de modelo.

### Intuición

Imagina que estás en una montaña nublada:
- No puedes ver el valle (el mínimo)
- Pero puedes sentir la pendiente bajo tus pies
- Si siempre das un paso en la dirección de la pendiente más pronunciada hacia abajo, eventualmente llegarás al valle

```
Función de Pérdida
       |
    ╱╲╲╲╲╲ (difícil, pedregoso)
   ╱  ╲╲╲╲╲
  ╱    ╲╲╲╲ ← paso grande
 ╱      ╲╲╲  (gradiente grande)
╱        ╲╲  ← paso pequeño
          ╲╲╲╲╲ (gradiente pequeño)
           ╲╲╲╲╲╲  ← llegamos
            ╲╲╲╲╲╲╲ (mínimo)
```

### El Algoritmo

1. Inicializar $\beta$ aleatoriamente
2. Repetir hasta convergencia:
   - Calcular el gradiente: $\nabla J = \frac{\partial J}{\partial \beta}$
   - Actualizar: $\beta := \beta - \alpha \nabla J$
   - $\alpha$ es la "tasa de aprendizaje" (qué tan grande es cada paso)

### Tasa de Aprendizaje

```
Tasa muy pequeña:      Tasa óptima:         Tasa muy grande:
    |                      |                    |
    |   convergencia        |  convergencia      |
    |   lenta               |  rápida            |  ¡diverge!
    |   _____               |___                 |  /\_/\_
    |  /                    |___               |_/\
    | /                     ___              /
    |/______               /___            /
         ↓                   ↓              ↓
```

### Formulación Matemática

Para regresión lineal con Error Cuadrático Medio:

$$J(\beta) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

El gradiente es:
$$\nabla J = -\frac{1}{n} X^T (y - \hat{y})$$

Y la actualización:
$$\beta := \beta + \alpha X^T (y - \hat{y})$$

### Implementación Conceptual

```python
import numpy as np

# Inicializar
beta = np.random.randn(n_features)
alpha = 0.01  # tasa de aprendizaje
n_iterations = 1000

# Agregar columna de 1s
X = np.column_stack([np.ones(len(X)), X])

for i in range(n_iterations):
    # Predicción
    y_pred = X @ beta
    
    # Calcular gradiente
    gradient = -X.T @ (y - y_pred) / len(y)
    
    # Actualizar pesos
    beta = beta - alpha * gradient
    
    # (Opcional) Imprimir pérdida
    if i % 100 == 0:
        loss = np.mean((y - y_pred) ** 2)
        print(f"Iteración {i}, Pérdida: {loss}")
```

## Regresión Polinomial

### El Problema: Relaciones No Lineales

A veces, la relación entre X e y no es lineal:

```
Datos lineales:        Datos polinomiales:
y                      y
|     ___              |      ___
|   _/                 |    /     \
|  /                   |   /       \
|_/______ x            |__/         \_ x
```

Regresión lineal no captura bien esta relación.

### La Solución

Crear nuevas características usando potencias de x:

$$\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \ldots$$

Esto se sigue llamando "regresión lineal" porque es lineal en los coeficientes $\beta$, aunque sea no lineal en x.

### Ejemplo

```python
# Si X es unidimensional
X_original = X

# Crear características polinomiales
X_poly = np.column_stack([
    X_original,
    X_original ** 2,
    X_original ** 3
])

# Ahora usar regresión lineal en X_poly
# es lo mismo que regresión polinomial
```

### Peligro: Overfitting

Polinomios de grado muy alto pueden provocar overfitting:

```
Grado 1:           Grado 3:           Grado 10:
(underfitting)     (bueno)            (overfitting)

y                  y                  y
|  ___             |     ___           |   /\   /\
| /                |   /     \         | _/  \_/  \
|/                 |  /       \        |/         \
____ x            ____ x             _____ x
```

## Regularización: Ridge y Lasso

### El Problema

Los polinomios de alto grado tienen coeficientes muy grandes que capturan el ruido. Queremos penalizar esto.

### Ridge Regression (L2)

Agregamos un término de penalización al MSE:

$$J(\beta) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \frac{\lambda}{2} \sum_{j=1}^{p} \beta_j^2$$

Esto **encoge** los coeficientes hacia cero, pero no los hace exactamente cero.

### Lasso Regression (L1)

Usa una penalización diferente:

$$J(\beta) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j|$$

Lasso puede hacer que algunos coeficientes sean **exactamente cero**, lo que es útil para **selección de características**.

### Comparación

| Aspecto | Ridge | Lasso |
|---------|-------|-------|
| **Penalización** | $\beta^2$ | $\|\beta\|$ |
| **Encoge hacia cero** | Suave | Abrupto |
| **Selección de features** | No | Sí |
| **Para interpretabilidad** | Peor | Mejor |

### El Parámetro $\lambda$

```
λ = 0:           λ = medio:      λ = grande:
Sobreajuste      Balance          Subajuste
(rojo = error)

Error en prueba
    |    ╱╲╲╲╲╲╲╲╲╲╲╲
    |   ╱  ╲╲╲╲╲╲╲╲
    |  ╱    ╲╲╲╲╲╲ ← óptimo aquí
    | ╱      ╲╲╲╲╲
    |╱________╲╲╲╲
    +---+---+---+---> λ
    0  pequeño grande
```

$\lambda$ se elige típicamente usando validación cruzada.

## Resumen

- **Regresión Lineal**: modelo simple pero poderoso
- **OLS**: solución exacta, pero costosa computacionalmente
- **Descenso de Gradiente**: método general, funciona para cualquier modelo
- **Regresión Polinomial**: captura relaciones no lineales
- **Regularización**: previene overfitting (Ridge para retención, Lasso para selección)

## Próximo

En el siguiente capítulo, aprenderemos a extender estas ideas a **Clasificación**, donde predecimos categorías en lugar de números continuos.
