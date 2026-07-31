# 3.1 Clustering

## Introducción

Clustering (agrupamiento) es la tarea de dividir datos en grupos (clusters) de manera que:
- Los puntos **dentro de un grupo sean similares** entre sí
- Los puntos en **grupos diferentes sean disimilares**

Es como encontrar familias naturales en tus datos sin que nadie te haya dicho cuáles son.

## K-Means

### El Algoritmo Más Simple

K-Means es el algoritmo de clustering más popular por su simplicidad y efectividad.

**Idea intuitiva**:
1. Escoger K centroides (centros de clusters) aleatorios
2. Asignar cada punto al centroide más cercano
3. Recalcular centroides como el promedio de puntos en cada cluster
4. Repetir hasta convergencia

### Visualización

```
Iteración 1:       Iteración 2:       Convergencia:
                                      
X X               X X X              X X X
X X    X ◆ X  →   X X ◆ X      →    X X ◆ X
   X X            ◆   X X            ◆   X X
     ◆  ◆        X ◆ X  X          X ◆ X  X
    X   X         X      X ◆        X      X ◆
  X ◆   X       X  X   X ◆        X  X   X ◆

◆ = centroide (centro del cluster)
```

### Algoritmo Formal

```
Entrada: datos X, número de clusters K

1. Inicializar K centroides aleatoriamente
2. Repetir hasta convergencia:
    a) Asignación: para cada punto, asignar al centroide más cercano
    b) Actualización: recalcular cada centroide como media de su cluster
3. Retornar: centroides finales y asignaciones
```

### Función Objetivo

K-Means intenta minimizar la **suma de distancias al cuadrado dentro de clusters**:

$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2$$

Donde:
- $C_k$ es el cluster k
- $\mu_k$ es el centroide del cluster k
- $||...||$ es la distancia euclidiana

### Implementación Conceptual

```python
import numpy as np

def kmeans(X, K, max_iterations=100):
    # 1. Inicializar centroides aleatoriamente
    centroids = X[np.random.choice(len(X), K, replace=False)]
    
    for iteration in range(max_iterations):
        # 2. Asignar cada punto al centroide más cercano
        distances = np.zeros((len(X), K))
        for k in range(K):
            distances[:, k] = np.linalg.norm(X - centroids[k], axis=1)
        assignments = np.argmin(distances, axis=1)
        
        # 3. Guardar centroides antiguos para verificar convergencia
        old_centroids = centroids.copy()
        
        # 4. Recalcular centroides
        for k in range(K):
            centroids[k] = X[assignments == k].mean(axis=0)
        
        # 5. Verificar convergencia
        if np.allclose(centroids, old_centroids):
            break
    
    return centroids, assignments
```

### Complejidad

- **Tiempo**: O(n * K * d * iteraciones) donde n = muestras, K = clusters, d = dimensiones
- **Espacio**: O(n * d)

Muy eficiente para datos grandes.

## Eligiendo K: El Método del Codo

### El Problema

¿Cómo saber cuántos clusters debe haber? K-Means requiere especificar K de antemano.

### La Solución: Elbow Method

Ejecutar K-Means para diferentes valores de K (1, 2, 3, ..., 10) y graficar la **inercia** (J):

```
Inercia
   |
 10|●
   |  \
   | 8 \●
   |     \●
   | 6    \
   |       \●
   | 4      \●
   |         \●(codo aquí)
   | 2        \●
   |           \●
   | 0          \●──────
   +---+---+---+---+---+-> K
   1   2   3   4   5   6
```

- Hasta el "codo": inercia decrece rápido (útil agregat K)
- Después del "codo": decrece lentamente (no agrega valor)
- Elegir K donde está el "codo"

### Interpretación

```
K=1: Todos en un grupo (alta inercia)
     
K=3: Grupos naturales emergen
     
K=10: Clusters innecesarios (overfitting)
```

### Implementación

```python
inertias = []
K_range = range(1, 11)

for K in K_range:
    centroids, assignments = kmeans(X, K)
    inertia = calculate_inertia(X, centroids, assignments)
    inertias.append(inertia)

# Graficar
import matplotlib.pyplot as plt
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K')
plt.ylabel('Inercia')
plt.show()
```

## Clustering Jerárquico

### Alternativa: Construir un Árbol

En lugar de decidir K de antemano, construimos una jerarquía de clusters:

```
Todos los datos
      |
   (fusionar)
      |
  Cluster 1    Cluster 2
      |            |
   (fusionar)   (fusionar)
      |            |
  C1.1 C1.2   C2.1 C2.2 C2.3
```

### Dendrograma

Una visualización de este árbol:

```
        ┌──────┐
        │      │
        │      │
      ┌─┤    ┌─┤
      │ │    │ │
    ┌─┤ │  ┌─┤ │
    │ │ │  │ │ │
    X Y Z  A B C
    1 2 3  4 5 6
```

Cortando horizontalmente en diferentes alturas → diferentes números de clusters.

### Métodos de Enlace (Linkage)

¿Cómo medimos distancia entre clusters?

**Single Linkage**: distancia entre los puntos más cercanos
```
Cluster A        Cluster B
    X ··· X
    
    ← mínima distancia
```
- Puede crear clusters largos y delgados

**Complete Linkage**: distancia entre los puntos más lejanos
```
Cluster A        Cluster B
    X           X
    
    ←── máxima distancia
    
    X           X
```
- Tiende a crear clusters compactos

**Average Linkage**: promedio de distancias
- Equilibrio entre ambos

**Ward Linkage**: minimiza incremento de varianza
- Generalmente produce buenos resultados

### Ventajas vs K-Means

| Aspecto | K-Means | Jerárquico |
|---------|---------|-----------|
| **K previo** | Requerido | No |
| **Visualización** | Difícil | Dendrograma |
| **Escalabilidad** | Excelente | Pobre (O(n²)) |
| **Clusters globales** | Mejores | Específicos |

## DBSCAN (Density-Based Spatial Clustering)

### Alternativa: Clustering por Densidad

A diferencia de K-Means que asume clusters esféricos, DBSCAN:
- Agrupa puntos en regiones densas
- Marca puntos aislados como outliers
- Descubre clusters de cualquier forma

### Conceptos Clave

**Vecindad (ε-neighborhood)**: todos los puntos dentro de distancia ε

```
Punto X:     Vecindad ε:
             
    o         ◆ = punto core (muchos vecinos)
  o X o       ○ = punto frontera (pocos vecinos)
    o o       ◇ = outlier (muy aislado)
```

**Punto Core**: tiene ≥ min_samples puntos en su ε-vecindad

**Punto Frontera**: no es core pero está en la vecindad de un core

**Outlier**: ni core ni frontera

### El Algoritmo

```
Para cada punto no visitado:
    Si es punto core:
        Crear nuevo cluster
        Agregar todos sus vecinos recursivamente
    Si no es core:
        Marcar como frontera u outlier
```

### Ventajas

- **Clusters arbitrarios**: no asume forma
- **Detecta outliers**: automáticamente
- **No requiere K**: parámetros ε y min_samples en su lugar
- **Escalable**: con índices espaciales

### Desventajas

- **Sensible a parámetros**: elegir ε y min_samples es difícil
- **Datos de alta dimensión**: distancias menos significativas
- **Densidades variadas**: difícil si clusters tienen densidades diferentes

### Comparación de Algoritmos

```
Datos Circulares:       Datos Complejos:      Datos con Outliers:

K-Means:                K-Means:             K-Means:
O O O                   X X   X              X X X
O O O    ✓              X   X   X (malo)     X X X (outlier etiquetado)
O O O                   X     X

DBSCAN:                 DBSCAN:              DBSCAN:
O O O                   X X   X              X X X
O O O    ✓              X   X   X ✓          X X X ✓ (outlier detectado)
O O O                   X     X
```

## Métrica de Evaluación: Silhueta

Sin etiquetas verdaderas, ¿qué tan bueno es nuestro clustering?

**Coeficiente de Silhueta**:

$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$

Donde:
- $a_i$ = distancia promedio a otros puntos en su cluster
- $b_i$ = distancia promedio al cluster más cercano

**Interpretación**:
- s = 1: punto está bien en su cluster
- s = 0: punto está en el borde
- s = -1: punto está en el cluster equivocado

Rango: [-1, 1], más alto es mejor.

## Resumen

- **K-Means**: simple, rápido, requiere K conocido
- **Método del Codo**: heurística para elegir K
- **Clustering Jerárquico**: dendrograma, mejor visualización
- **DBSCAN**: detecta formas arbitrarias y outliers
- **Silhueta**: métrica para evaluar calidad sin etiquetas

---

**Siguiente**: En 3.2 aprenderemos a reducir dimensionalidad para simplificar y visualizar datos complejos.
