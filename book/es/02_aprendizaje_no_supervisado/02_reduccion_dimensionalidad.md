# 3.2 Reducción de Dimensionalidad

## Introducción

Muchos datasets tienen cientos o miles de características. Trabajar con tantas dimensiones es:
- **Computacionalmente costoso**: algoritmos más lentos
- **Difícil de visualizar**: no podemos ver datos de 100 dimensiones
- **Ruidoso**: muchas características irrelevantes
- **Maldición de la dimensionalidad**: modelos generalizan peor

La reducción de dimensionalidad **comprime datos complejos** en pocas dimensiones manteniendo información esencial.

## PCA: Principal Component Analysis

### Intuición

Imagina que tienes datos en 3D. Si la mayoría de la variación ocurre en 2 direcciones principales, puedes proyectar a esas 2 dimensiones sin perder mucha información.

```
Datos 3D:            Proyección 2D:
    z                    y
    |      • •           |    • •
    |    •   •           |  •   •
    |  •       •    →    |•       •
    +---x    •           +---x
       y  •
```

### Componentes Principales

Los "componentes principales" son direcciones donde los datos varían más.

- **PC1**: dirección de máxima varianza
- **PC2**: dirección de segunda mayor varianza, ortogonal a PC1
- **PC3**: tercera dirección, ortogonal a PC1 y PC2
- ...

```
Varianza en diferentes direcciones:

Alta varianza     PC1 (máxima)
    →
   /
  /            PC2 (ortogonal)
                →
```

### Algoritmo Formal

1. **Centrar datos**: restar la media
2. **Calcular matriz de covarianza**: qué características varían juntas
3. **Descomposición en valores propios**: encontrar direcciones principales
4. **Proyectar**: multiplicar datos por los vectores propios

### Varianza Explicada

Cada componente explica una cierta porcentaje de la varianza total:

```
Varianza explicada acumulada:
    |
 100|────────────────●
    |              ╱
  50|           ●
    |        ╱
  10|     ●
    |
    +---+---+---+---> Número de componentes
    0   5   10  15
```

**Decisión**: ¿cuántos componentes mantener?
- Mantener 95% de varianza explicada es típico
- Reduce dimensionalidad pero conserva estructura

### Implementación Conceptual

```python
import numpy as np

def pca(X, n_components):
    # 1. Centrar datos
    X_centered = X - X.mean(axis=0)
    
    # 2. Matriz de covarianza
    cov_matrix = np.cov(X_centered.T)
    
    # 3. Descomposición en valores propios
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # 4. Ordenar por valor propio (varianza explicada)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 5. Seleccionar top n_components
    components = eigenvectors[:, :n_components]
    
    # 6. Proyectar datos
    X_reduced = X_centered @ components
    
    return X_reduced, components, eigenvalues
```

### Interpretación

Aunque PCA reduce dimensiones, las nuevas características (componentes) son **combinaciones de las originales**:

```
PC1 = 0.7 * Feature1 + 0.3 * Feature2 - 0.1 * Feature3
PC2 = 0.2 * Feature1 - 0.8 * Feature2 + 0.5 * Feature3
```

Es difícil interpretar qué significa cada componente.

### Ventajas y Desventajas

| Aspecto | PCA |
|---------|-----|
| **Rápido** | ✓✓ |
| **Interpretable** | ✗ (componentes son mezclas) |
| **Lineal** | Solo captura relaciones lineales |
| **Óptimo** | Maximiza varianza (no siempre lo correcto) |

## t-SNE (t-Distributed Stochastic Neighbor Embedding)

### Problema con PCA

PCA es lineal. Para datos complejos con estructura no lineal, falla:

```
Datos en espiral       PCA (fallida):       t-SNE (buena):

    •••               ••     •••            •••
  ••   ••            •   ••   •             •••••
 •       •           •••     •            •      •
  •••••••••    →    • •• •• •       →    •        •
   •     •             •••                 •     •
    •   •              •••
     •••
```

### Idea: Conservar Similaridades Locales

En lugar de conservar varianza global, t-SNE intenta:
- Puntos similares en espacio alto → cercanos en espacio bajo
- Puntos disímiles en espacio alto → lejanos en espacio bajo

### Propiedades

- **No lineal**: puede descubrir estructuras complejas
- **Preserva estructura local**: lo que ves es real localmente
- **Pero**: distancias globales no son significativas

```
❌ NO ES CORRECTO decir:
"Los dos clusters están a distancia X"
"El cluster A es más grande que B"

✓ ES CORRECTO decir:
"Los puntos A y B están en el mismo vecindario"
"Hay dos grupos distintos"
```

### Parámetros Clave

- **perplexity**: número de vecinos a considerar (típicamente 30-50)
- **learning_rate**: velocidad de optimización
- **n_iterations**: cuántas iteraciones ejecutar

Estos parámetros afectan el resultado final significativamente.

### Interpretación

```
t-SNE 2D de dataset MNIST (dígitos 0-9):

0 0 0 0
0 0 1 1 1
0   1 1 1 1 1
  2 2 2 1
 2 2 2 3 3 3 3 3
 2 2 3 3 3 3
4 4 4 4 3
4 4 4 4 4
5 5 5 5
5 5 5 5
6 6 6 6 6 6
6 6 6 6
7 7 7
7 7 7 7 7
8 8 8 8
8 8 8 8 8 8
9 9 9 9
9 9 9 9
```

Vemos dígitos similares agrupados naturalmente.

## UMAP (Uniform Manifold Approximation and Projection)

### Mejora sobre t-SNE

UMAP es similar a t-SNE pero con ventajas:
- **Más rápido**: especialmente para datasets grandes
- **Preserva escala global mejor**: distancias globales más significativas
- **Más estable**: resultados reproducibles con mismos parámetros

### Cuándo Usar Qué

| Situación | Recomendar |
|-----------|-----------|
| **Visualizar datos complejos** | t-SNE o UMAP |
| **Dataset muy grande (>10k)** | UMAP (más rápido) |
| **Necesitas interpretación global** | PCA |
| **Necesitas velocidad** | PCA |
| **Dataset pequeño (<1k)** | t-SNE |

## Aplicaciones de Reducción de Dimensionalidad

### 1. Compresión de Datos
```
Imágenes 28×28 = 784 dimensiones
PCA a 50 componentes = 93% varianza explicada
Compresión 15x con mínima pérdida visual
```

### 2. Visualización para Exploración
```
Datos de 100 dimensiones
Aplicar t-SNE a 2D
Visualizar y detectar clusters naturales
```

### 3. Preprocesamiento
```
Datos ruidosos 1000D
Aplicar PCA mantener 95% varianza
Reduce ruido y acelera algoritmos posteriores
```

### 4. Feature Engineering
```
Crear nuevas características usando componentes
Como inputs a modelos posteriores
```

## Maldición de la Dimensionalidad

Entender por qué reducción es importante:

```
En 1D (línea):
Puntos aleatorios: X - X - - X - X -
Unos están cercanos, otros lejanos

En 2D (plano):
Puntos aleatorios: X - - - X - X - -
                   - - - - - - - - -
                   X - - - - X - - -
Ahora más separados

En 1000D (hipercubo):
Puntos aleatorios: CASI TODOS estabAn equidistantes
Distancia entre puntos aleatorios ≈ constante
No hay información de proximidad
```

**Conclusión**: en alta dimensión, todos los puntos se vuelven casi equidistantes. La noción de "cercano" desaparece.

## Resumen

- **PCA**: rápido, lineal, interpatable características originales
- **t-SNE**: visualización excelente, no lineal, lento para datos grandes
- **UMAP**: balance entre PCA y t-SNE
- **Varianza explicada**: métrica para elegir número de componentes
- **Maldición de la dimensionalidad**: motiva reducción
- **Aplicaciones**: compresión, visualización, preprocesamiento

---

**Siguiente**: En 3.3 aprenderemos a detectar anomalías, el último tema en aprendizaje no supervisado.
