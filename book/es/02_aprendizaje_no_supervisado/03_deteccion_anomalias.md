# 3.3 Detección de Anomalías

## Introducción

Las anomalías son puntos de datos que se desvían significativamente del comportamiento "normal".

```
Datos Normales:        Con Anomalías:
  X X                    X X
X   X         →        X   X
  X X                    X X
                           ◆ (anomalía)
```

La detección de anomalías tiene aplicaciones críticas:
- **Seguridad**: fraude en transacciones bancarias
- **Mantenimiento**: fallos incipientes en maquinaria
- **Biología**: mutaciones genéticas raras
- **Astronomía**: eventos cósmicos inusuales

## Conceptos Fundamentales

### ¿Qué es una Anomalía?

Una observación que:
1. Es rara (ocurre infrequentemente)
2. Es diferente del patrón normal
3. Sugiere un proceso generativo diferente

```
Distribución Normal + Anomalías:

Densidad
    |
    |●●●●●
    |●   ●
    |●   ●      ◆ anomalía 1
    |●   ● ◆
    |●___●   ◆ anomalía 2
    +--+--+--+---> Valor
```

### Tipos de Anomalías

#### Punto (Point Anomaly)
Un punto individual anómalo en el contexto de todos los datos.

```
[normal, normal, anomalía, normal, normal]
```

#### Contextual (Contextual Anomaly)
Normal en general, pero anómalo en este contexto específico.

```
Temperatura en Salamanca:
Enero: 2°C   (normal)
Julio: 35°C  (normal)
Diciembre: 25°C (¡anomalía!)
```

#### Colectivo (Collective Anomaly)
Una colección de puntos es anómala, aunque individualmente sean normales.

```
Patrón de fraude: transacción normal + transferencia normal
Juntas forman patrón de lavado de dinero
```

## Métodos de Detección

### 1. Estadístico: Gaussiana Univariada

Asumir que datos normales siguen distribución normal:

```
Densidad
    |
    |      ╱╲
    |    ╱    ╲
    |  ╱        ╲    ◆ (fuera de límites)
    |╱            ╲__
    +---+---+---+---+----> x
     -3σ -2σ -σ  0  +σ
```

Algoritmo:
1. Estimar media μ y desviación σ de datos normales
2. Para nuevo punto x, calcular probabilidad P(x)
3. Si P(x) < umbral, es anomalía

$$P(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$$

**Regla práctica**: puntos a ±3σ de la media son anomalías (99.7% de datos).

**Limitaciones**: solo funciona bien para una dimensión. En alta dimensión, la mayoría de puntos están "en la cola".

### 2. Basado en Distancia: Aislamiento por Proximidad

Si un punto está muy alejado de sus vecinos → anomalía.

```
Punto normal:       Anomalía:
  o o               o   o
o   o      →      o       o
  o o              
      o ◆                 ◆
   (cerca)          (aislado)
```

Métodos:
- **K-Nearest Neighbors (k-NN)**: punto con vecinos lejanos
- **LOF (Local Outlier Factor)**: compara densidad local con vecinos
- **Isolation Forest**: árbol que aísla puntos anómalos

### 3. Aislamiento (Isolation Forest)

**Idea**: los puntos anómalos son más fáciles de aislar.

```
Partición 1: Dividir por feature_1
    [Normal] | [Normal, Anomalía]
    
Partición 2: Dividir por feature_2 (en rama con anomalía)
    [Normal] | [Anomalía]
    
Anomalía aislada en 2 particiones
Normal requeriría 10+ particiones
```

**Ventajas**:
- Eficiente: O(n log n)
- No paramétrico
- Funciona bien en alta dimensión
- Naturalmente detecta puntos anómalos

**Desventajas**:
- Menos interpretable
- Sensible a randomización

### 4. Basado en Densidad: Local Outlier Factor (LOF)

Compara densidad local con densidad de vecinos:

```
Región densa:       Región rara:
X X X X              X   X   X
X   X      densidad  X       X (baja densidad)
X X X X      alta    X   X   X
```

LOF = 1: densidad similar a vecinos (normal)
LOF > 1: menos denso que vecinos (anomalía)

$$\text{LOF}(p) = \frac{\text{densidad media de vecinos}}{\text{densidad de } p}$$

### 5. Autoencoders

Red neuronal que:
1. Comprime datos en representación de bajo rango
2. Reconstruye desde esa representación

```
Entrada → Compresión → Reconstrucción → Salida
  x           z(100D)      x'
             (10D)
             
Error de reconstrucción = ||x - x'||
Si error > umbral → anomalía
```

**Intuición**: 
- Datos normales: reconstrucción buena
- Datos anómalos: reconstrucción pobre (nunca los vio)

**Ventaja**: aprende qué es "normal" de los datos

## Evaluación de Anomalías

### El Desafío

Típicamente hay muy pocas anomalías verdaderas:

```
1000 transacciones: 999 normales, 1 fraude
Exactitud de decir "todo es normal": 99.9%
Pero eso es inútil, no detectó el fraude
```

### Métricas Adecuadas

**Recall (Sensibilidad)**: 
$$\text{Recall} = \frac{\text{Anomalías detectadas}}{\text{Todas las anomalías}}$$

Crítico: queremos encontrar todas las anomalías.

**Precisión**:
$$\text{Precisión} = \frac{\text{Anomalías correctas}}{\text{Predichas como anomalía}}$$

Si es muy bajo, investigar muchos falsos positivos.

**F1-Score**: balance entre ambas.

### Curva Precision-Recall

```
Precisión
    |
  1 |●
    |  ●
    |    ●●  ← óptimo
    |        ●●●
    |            ●●
  0 |________________●----> Recall
    0            0.5    1
```

A diferencia de ROC, esta es más informativa para clases desbalanceadas.

## Aplicaciones Reales

### Detección de Fraude Bancario

```
Datos normales:
- Transacciones pequeñas (< 500€)
- Horarios de trabajo
- Ubicación consistente

Anomalías:
- Transacción grande (50000€) a las 2 AM
- Ubicación cambió 1000km en 2 horas
- Múltiples transacciones rechazadas luego aceptada
```

### Mantenimiento Predictivo

```
Sensor de vibración en máquina:
- Normal: 10-20 Hz
- Anomalía: picos > 50 Hz
→ Indicador de fallo incipiente

Acción: mantenimiento preventivo antes de fallo
Ahorro: evita paradas inesperadas
```

### Detección de Intrusiones

```
Flujo de red normal:
- Pocos intentos de conexión por segundo
- Puertos estándares (80, 443)
- Volumen consistente

Anomalía:
- Escaneo de puertos (múltiples puertos en segundos)
- Volumen explosivo de tráfico
- Puertos inusuales

Acción: bloquear conexión, alertar
```

## Implementación Conceptual: Isolation Forest

```python
import numpy as np

class IsolationForest:
    def __init__(self, n_trees=100, max_depth=20):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
    
    def fit(self, X):
        for _ in range(self.n_trees):
            tree = self._build_tree(X, depth=0)
            self.trees.append(tree)
    
    def _build_tree(self, X, depth):
        if depth >= self.max_depth or len(X) <= 1:
            return {'type': 'leaf', 'size': len(X)}
        
        # Elegir feature aleatorio
        feature = np.random.randint(0, X.shape[1])
        
        # Elegir valor de split aleatorio
        split_value = np.random.uniform(X[:, feature].min(), 
                                        X[:, feature].max())
        
        # Dividir datos
        left_idx = X[:, feature] < split_value
        right_idx = ~left_idx
        
        return {
            'type': 'internal',
            'feature': feature,
            'split': split_value,
            'left': self._build_tree(X[left_idx], depth + 1),
            'right': self._build_tree(X[right_idx], depth + 1)
        }
    
    def predict(self, X):
        anomaly_scores = []
        for x in X:
            # Calcular profundidad promedio en árboles
            depths = [self._traverse(x, tree, 0) 
                     for tree in self.trees]
            avg_depth = np.mean(depths)
            anomaly_scores.append(avg_depth)
        
        # Normalizar: bajo score = anomalía
        return np.array(anomaly_scores)
    
    def _traverse(self, x, node, depth):
        if node['type'] == 'leaf':
            return depth
        
        if x[node['feature']] < node['split']:
            return self._traverse(x, node['left'], depth + 1)
        else:
            return self._traverse(x, node['right'], depth + 1)
```

## Comparación de Métodos

| Método | Complejidad | Lineal | Interpretable | Alto-D |
|--------|------------|--------|---------------|--------|
| **Gaussiana** | O(n) | ✓ | ✓✓ | ✗ |
| **k-NN** | O(n²) | ✗ | ○ | ✗ |
| **LOF** | O(n²) | ✗ | ○ | ✗ |
| **Isolation Forest** | O(n log n) | ✗ | ○ | ✓ |
| **Autoencoder** | O(n) | ✗ | ✗ | ✓✓ |

## Resumen

- **Anomalías**: eventos raros, contextuales o colectivos
- **Métodos simples**: gaussiana para 1D, k-NN para distancia
- **Métodos avanzados**: Isolation Forest, Autoencoders
- **Evaluación**: Recall más importante que Exactitud
- **Aplicaciones**: fraude, mantenimiento, seguridad

## Reflexión Final

La detección de anomalías es tanto **arte como ciencia**:
- La ciencia: algoritmos probados
- El arte: entender tu dominio para definir qué es "anómalo"

Una anomalía estadística puede ser perfecto en tu negocio. Una anomalía empresarial puede ser perfectamente normal estadísticamente.

---

¡Felicidades! Has completado el curso de Fundamentos e Introducción al Aprendizaje Automático. Ahora tienes bases sólidas para explorar temas más avanzados como **aprendizaje profundo**, **procesamiento de lenguaje natural**, o **visión por computadora**.

**Próximos pasos recomendados**:
1. Practicar con datasets reales (Kaggle, UCI)
2. Implementar algoritmos desde cero
3. Leer papers académicos
4. Contribuir a proyectos open-source
