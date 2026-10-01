# 3.1 *Clustering*

## Introducción

El aprendizaje no supervisado representa un paradigma fundamental de la inteligencia artificial, caracterizado por la ausencia de etiquetas o variables objetivo que guíen el proceso de entrenamiento. En este bloque, el sistema no recibe un criterio de "verdad fundamental" (*ground-truth*), sino que debe explorar las relaciones matemáticas intrínsecas de los datos para revelar su estructura oculta.

Dentro de este paradigma, el ***clustering*** (o agrupamiento) es la técnica más extendida, orientada a particionar un conjunto de datos en subgrupos cuyos miembros compartan una alta similitud interna y, al mismo tiempo, presenten una alta diferenciación respecto a los miembros de otros subgrupos.

## Algoritmos de *clustering*

Existen diferentes algoritmos de ***clustering***. A continuación, se explican los mas relevantes actualmente.

### Algoritmo $K$-*means*

El $K$-*means* {cite:p}`macqueen1967kmeans` es un algoritmo de agrupación que realiza las particiones basada en centroides. El algoritmo es una técnica de agrupamiento iterativo que busca particionar un conjunto de observaciones en $K$ *clusters* distintos. Es un enfoque geométrico basado en el concepto de centroides, que actúan como los centros de gravedad de cada grupo.

La {numref}`fig-kmeans-flow` resume el ciclo del algoritmo: tras asignar cada muestra a su centroide y calcular la inercia (WCSS), se actualizan los centroides y el proceso se repite hasta alcanzar la convergencia.

```{figure} ../../_static/generated/diagrams/es/02_aprendizaje_no_supervisado_01_clustering_01.svg
:name: fig-kmeans-flow
:alt: Diagrama de flujo del algoritmo K-means con las etapas de asignación a centroides, cálculo de la inercia y actualización de centroides en bucle hasta la convergencia
:width: 100%
:align: center

Ciclo iterativo de $K$-*means*.
```

#### Mecánica algorítmica y convergencia

La meta matemática de $K$-*means* es minimizar la inercia intracluster, formalmente conocida como la suma de los cuadrados dentro del  *cluster*  (WCSS - *Within-cluster Sum of Squares*). El proceso de optimización se divide en los siguientes pasos iterativos:

1. **Inicialización**: se definen $K$ puntos en el espacio de características para que actúen como centroides iniciales.
2. **Paso de asignación** (expectación): cada muestra del *dataset*  se asigna al centroide más cercano utilizando una métrica de distancia, típicamente la distancia Euclidiana: 
    $d(x, y) = \|x - y\|_2 = \sqrt{\sum_{i=1}^{D} (x_i - y_i)^2}$
3. **Paso de actualización** (maximización): se recalcula la posición de cada centroide como el promedio matemático (la media aritmética) de todas las muestras asignadas a dicho cluster.
4. **Convergencia**: los pasos 2 y 3 se repiten de forma iterativa hasta que los centroides no cambien su posición significativamente o las asignaciones de las muestras permanezcan estables.

La {numref}`fig-kmeans-process` muestra las tres etapas sobre un conjunto de datos con tres grupos naturales: los centroides (marcados con X) parten de posiciones aleatorias y convergen hacia el centro de cada grupo tras pocas iteraciones.

```{figure} ../../_static/generated/figures/es/kmeans_process.png
:name: fig-kmeans-process
:alt: Tres paneles mostrando la inicialización, una iteración intermedia y la convergencia final del algoritmo K-means
:width: 100%
:align: center

Proceso iterativo de $K$-*means*: de la inicialización a la convergencia.
```

##### El problema de los mínimos locales e inicialización ($K$-*means*++)

$K$-*means* es un algoritmo heurístico y, por tanto, es altamente sensible a la inicialización aleatoria de sus centroides. Si los centroides iniciales se colocan en regiones desfavorables del espacio de características, el algoritmo puede converger en mínimos locales subóptimos.

Para mitigar esta vulnerabilidad, se utiliza la inicialización $K$-*means*++:

- El primer centroide se selecciona de manera uniformemente aleatoria entre los puntos de datos.
- Los centroides subsiguientes se eligen de forma probabilística, donde la probabilidad de seleccionar un punto como nuevo centroide es proporcional al cuadrado de la distancia al centroide más cercano ya existente: 
    $P(x) \propto D(x)^2$
 
Este método de siembra inteligente asegura que los centroides iniciales estén ampliamente distribuidos por todo el espacio de características, acelerando la convergencia y mejorando la calidad del agrupamiento final

#### Selección de la complejidad del modelo

##### Método del codo (*elbow method*)

Dado que el número de *clusters* $K$ es un hiperparámetro que debe ser definido por el usuario antes del entrenamiento, se requiere un protocolo científico para su selección.

El método del codo consiste en graficar el valor del WCSS en función de diferentes valores de $K$:

- A medida que $K$ aumenta, la WCSS disminuye de manera natural porque los *clusters* se vuelven más pequeños y los puntos están más cerca de sus respectivos centroides.
- El valor óptimo de $K$ se localiza en el punto de inflexión de la curva (el "codo"), donde el descenso de la inercia deja de ser abrupto y se vuelve lineal. Esto representa un equilibrio óptimo entre la complejidad del modelo y la cohesión de los grupos.

La {numref}`fig-elbow` muestra un ejemplo típico: la inercia cae bruscamente hasta $K=3$ y después apenas mejora, señalando que 3 es el número de clusters más razonable para estos datos.

```{figure} ../../_static/generated/figures/es/elbow_method.png
:name: fig-elbow
:alt: Gráfica de la inercia (WCSS) frente al número de clusters K, con el codo marcado en K=3
:width: 70%
:align: center

Método del codo: la inercia deja de disminuir bruscamente a partir de $K=3$.
```

### *Clustering* jerárquico: aglomerativo y dendrogramas

El *clustering* jerárquico agrupa datos sin la necesidad de definir de antemano el número de grupos $K$. Su principal ventaja es que genera una estructura jerárquica de agrupamiento que puede visualizarse fácilmente.

#### Enfoque aglomerativo (*bottom-up*)

El método más común es el jerárquico aglomerativo, que opera de abajo hacia arriba. El algoritmo comienza tratando a cada observación individual como un único  *cluster*  independiente. En cada paso iterativo, se calculan las distancias entre todos los *clusters* y se fusionan los dos grupos más cercanos. Este proceso de fusión se repite de manera sucesiva hasta que todas las muestras quedan integradas en un único  *cluster*  global.

#### Dendrogramas e interpretación de cortes

La jerarquía de fusiones se representa gráficamente mediante un diagrama de árbol llamado dendrograma. El eje vertical del dendrograma representa la distancia de fusión (o disimilitud cophenética) entre los subgrupos.

El eje horizontal organiza las observaciones individuales de forma que las ramas similares queden adyacentes. El usuario puede determinar el número final de *clusters* realizando un corte horizontal en el dendrograma a una altura de disimilitud específica. La cantidad de líneas verticales interceptadas por el corte define el número de grupos resultantes, ofreciendo una flexibilidad interpretativa que algoritmos rígidos como $K$-*means* no poseen.

La {numref}`fig-dendrogram` muestra un dendrograma real generado sobre datos con tres grupos: la línea roja discontinua marca un corte que produce exactamente 3 clusters (uno por color).

```{figure} ../../_static/generated/figures/es/dendrogram.png
:name: fig-dendrogram
:alt: Dendrograma de clustering jerárquico con tres ramas coloreadas y una línea de corte horizontal que produce tres clusters
:width: 80%
:align: center

Dendrograma de *clustering* jerárquico con un corte que produce 3 clusters.
```

#### Criterios de enlace (*linkage methods*)

Para determinar qué *clusters* deben fusionarse en cada paso, se debe definir cómo medir la distancia o disimilitud entre grupos que contienen múltiples observaciones. Los principales métodos de enlace son:

- ***Complete linkage*** (enlace completo): calcula la distancia máxima entre cualquier miembro del primer  *cluster*  y cualquier miembro del segundo cluster: 
    $D(A, B) = \max_{x \in A, y \in B} d(x, y)$
Produce dendrogramas compactos y *clusters* de formas esféricas bien balanceadas.

- ***Single linkage*** (enlace simple): calcula la distancia mínima entre cualquier miembro del primer  *cluster*  y cualquier miembro del segundo cluster: 
    $D(A, B) = \min_{x \in A, y \in B} d(x, y)$
Es sensible al ruido y puede generar el efecto de encadenamiento (*chaining*), donde los *clusters* se fusionan de forma alargada y difusa.

- ***Average linkage*** (enlace promedio): calcula el promedio de todas las distancias entre cada punto del primer  *cluster*  y cada punto del segundo:
    $D(A, B) = \frac{1}{|A||B|} \sum_{x \in A} \sum_{y \in B} d(x, y)$
Es un criterio robusto que equilibra la estabilidad del enlace completo con la tolerancia del enlace simple

La {numref}`fig-islr-nci60-dendrogram` compara los tres criterios de enlace sobre datos reales de expresión génica del conjunto *NCI60* (líneas celulares de cáncer): nótese cómo el enlace simple (abajo) produce el efecto de encadenamiento característico, mientras que el enlace completo y el promedio (arriba y centro) generan agrupamientos más equilibrados.

```{figure} ../../_static/book_figures/islr_fig10_17_nci60_dendrogram.png
:name: fig-islr-nci60-dendrogram
:alt: Tres dendrogramas del conjunto de datos NCI60 usando enlace completo, promedio y simple, tomados del libro An Introduction to Statistical Learning
:width: 85%
:align: center

Clustering jerárquico del conjunto de datos *NCI60* con enlace completo, promedio y simple.
Fuente: James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*, Figura 10.17. Springer. Libro de libre distribución para uso educativo (statlearning.com).
```

### DBSCAN: agrupamiento basado en densidad

El algoritmo DBSCAN (*Density-Based Spatial clustering of Applications with Noise*) {cite:p}`ester1996dbscan` ofrece un enfoque radicalmente diferente a $K$-*means* y al **clustering** jerárquico al definir los grupos en función de la densidad local de los datos en el espacio de características. Esto le permite descubrir *clusters* de formas geométricas arbitrarias y aislar de forma natural las muestras de ruido

#### Fundamentos y parámetros críticos

DBSCAN requiere la configuración de dos hiperparámetros esenciales que definen la densidad local del entorno: 

- **$\epsilon$ (Epsilon)**: el radio de vecindad que define el entorno alrededor de cualquier punto.
- **MinPts** (muestras mínimas): el número mínimo de puntos que deben existir dentro del radio $\epsilon$ para considerar que esa región es densa.

#### Clasificación de puntos en DBSCAN

Durante su ejecución, DBSCAN examina cada punto del *dataset*  y lo clasifica en una de tres categorías exclusivas según las condiciones de densidad de su vecindad:

- ***Core points*** (puntos núcleo): un punto es catalogado como núcleo si su vecindad de radio $\epsilon$ contiene al menos un número de muestras igual o superior a MinPts.
- ***Border points*** (puntos frontera): son aquellos puntos que no cumplen con el requisito de densidad mínima de MinPts para ser considerados núcleo, pero residen dentro de la vecindad $\epsilon$ de algún punto que sí es núcleo.
- ***Noise points*** (puntos de ruido / *outliers*): cualquier punto que no es clasificado como núcleo ni como frontera. Estos puntos se consideran anomalías o valores atípicos y no son asignados a ningún cluster.

La {numref}`fig-dbscan-points` ilustra los tres tipos con $\epsilon = 1$ y MinPts $= 4$: los puntos azules tienen al menos 4 vecinos dentro de su círculo (núcleo); los naranjas caen dentro del círculo de un punto núcleo pero no alcanzan MinPts por sí mismos (frontera); y la cruz roja no tiene ningún punto núcleo cerca (ruido).

```{figure} ../../_static/generated/figures/es/dbscan_point_types.png
:name: fig-dbscan-points
:alt: Diagrama con puntos núcleo en azul rodeados de círculos de radio epsilon, puntos frontera en naranja dentro de esos círculos y un punto de ruido en rojo aislado
:width: 70%
:align: center

Clasificación de puntos en DBSCAN: núcleo, frontera y ruido.
```

#### Ventajas sobre métodos basados en distancia

- **Robustez ante el ruido**: a diferencia de $K$-*means*, que obliga a cada muestra a pertenecer a un  *cluster*  (desplazando artificialmente los centroides ante valores atípicos), DBSCAN identifica y aísla el ruido de forma nativa.
- **Flexibilidad geométrica**: no asume que los *clusters* tienen que ser esféricos; puede descubrir *clusters* anidados, lineales o con formas complejas de baja dimensión dentro de espacios de alta dimensión.

La {numref}`fig-dbscan-vs-kmeans` compara ambos algoritmos sobre datos con forma no convexa: $K$-*means* corta el grupo por la mitad de forma arbitraria, mientras que DBSCAN respeta la forma real definida por la densidad.

```{figure} ../../_static/generated/figures/es/dbscan_vs_kmeans.png
:name: fig-dbscan-vs-kmeans
:alt: Comparación lado a lado de K-means y DBSCAN sobre datos con forma de luna, mostrando cómo K-means falla al dividir la forma no convexa mientras DBSCAN la mantiene íntegra
:width: 100%
:align: center

$K$-*means* frente a DBSCAN en datos con formas no convexas.
```

## Técnicas de evaluación en **clustering**

La evaluación de las estructuras de agrupamiento es uno de los campos más complejos y debatidos del aprendizaje no supervisado, ya que, al no contar con etiquetas de clase o una "verdad fundamental" (*ground-truth*) durante el entrenamiento, no existe un criterio único y objetivo de éxito. La calidad de un agrupamiento a menudo se mide de forma subjetiva en función de la utilidad del modelo para el usuario final en su dominio de aplicación.

Sin embargo, para aproximar este análisis de manera rigurosa, la literatura científica ha consolidado métodos de evaluación divididos en cuatro  grandes enfoques: (i) métricas geométricas internas, (ii) criterios basados en la teoría de la información, (iii) métricas y enfoques de información unificados y (iv) validaciones externas supervisadas.

### Métricas de evaluación interna y estructura geométrica

Analizan la disposición espacial de los puntos en el espacio de características para medir dos propiedades deseables: que los *clusters* sean lo más compactos posibles (baja varianza interna) y que estén bien separados entre sí.

#### Coeficiente y *score* de silueta (*silhouette score*)

El coeficiente de silueta ($sc$) {cite:p}`rousseeuw1987silhouette` evalúa la calidad de la asignación de cada muestra de forma individual, sirviendo de diagnóstico para agrupamientos predominantemente esféricos. Para una instancia $i$, se calcula como:

$sc(i) = \frac{b_i - a_i}{\max(a_i, b_i)}$ 

Donde:

- $a_i$ (**cohesión**): es la distancia promedio (usualmente euclídea) entre la instancia $i$ y todos los demás puntos que pertenecen a su mismo cluster. Queremos que este valor sea lo más bajo posible.
- $b_i$ (**separación**): es la distancia promedio entre la instancia $i$ y todas las muestras del *cluster* vecino más cercano (el *cluster* que minimice este valor de distancia promedio, excluyendo el propio). Queremos que este valor sea lo más alto posible.

##### Interpretación de los resultados

- El coeficiente varía estrictamente en el rango de $[-1, 1]$.
- Un valor cercano a $+1$ indica que la muestra está en el interior de su propio grupo y muy alejada de las fronteras de otros clusters.
- Un valor cercano a $0$ denota que la muestra se sitúa sobre la frontera de decisión entre dos clusters.
- Un valor cercano a $-1$ sugiere fuertemente que la muestra ha sido asignada al grupo incorrecto.

El *score* de silueta global es el promedio de los coeficientes de todas las instancias del conjunto de datos. En un diagrama de silueta, los coeficientes se ordenan de mayor a menor y se grafican por cluster. La altura de cada silueta (forma de cuchillo) indica el tamaño del *cluster* y el ancho representa el coeficiente individual de sus puntos. Esto permite identificar visualmente si algún *cluster* es excesivamente grande o si tiene demasiadas muestras por debajo del *score* promedio general (representado por una línea vertical discontinua), lo cual delata agrupamientos de baja calidad.

La {numref}`fig-silhouette` muestra un diagrama de silueta real con tres clusters: la mayoría de las muestras del cluster 1 (rojo) y 0 (azul) superan la silueta promedio, mientras que el cluster 2 (verde) es más heterogéneo.

```{figure} ../../_static/generated/figures/es/silhouette_diagram.png
:name: fig-silhouette
:alt: Diagrama de silueta mostrando los coeficientes de silueta agrupados por cluster, con una línea vertical marcando la silueta promedio
:width: 70%
:align: center

Diagrama de silueta: cada "cuchillo" representa un cluster y su calidad de agrupamiento.
```

#### Estadístico *gap* (*gap statistic*)

Propuesto por Tibshirani et al. {cite:p}`tibshirani2001gapstatistic`, el estadístico *gap* formaliza matemáticamente la búsqueda del número óptimo de *clusters* ($K$) superando la vaguedad visual del método heurístico del codo (*elbow method*).

Compara la curva del logaritmo de la inercia observada de tus datos ($\log W_K$) con la esperanza matemática de la inercia calculada sobre múltiples muestras generadas artificialmente con una distribución uniforme (sin estructura de agrupamiento o hipótesis nula) en el hiperrectángulo que encierra a los datos reales:

$\text{Gap}(K) = E^*[\log W_K] - \log W_K$

El método estima el $K$ óptimo maximizando este "*gap*" o brecha. Matemáticamente, la regla formal selecciona el menor $K$ tal que:

$K^* = \text{argmin}_K \{ K \mid \text{Gap}(K) \ge \text{Gap}(K+1) - s'_{K+1} \}$

Donde $s'_{K+1} = s_{K+1}\sqrt{1 + 1/B}$ representa la desviación estándar de la simulación ajustada por el número de réplicas artificiales $B$. Su mayor ventaja es que puede determinar si el número óptimo de *clusters* es igual a 1 (es decir, que los datos no tienen *clusters* naturales), un escenario donde la inercia o el coeficiente de silueta fallan.

#### Coeficiente de correlación cofenética

Esta técnica evalúa la calidad de la estructura jerárquica impuesta por el *clustering* jerárquico. Mide la correlación de Pearson entre la matriz de disimilitudes iniciales $\{d_{ii'}\}$ de las observaciones y sus correspondientes disimilitudes cofenéticas $\{C_{ii'}\}$ obtenidas a partir de las alturas de fusión en el dendrograma.

La distancia cofenética $C_{ii'}$ es la altura exacta del enlace en el dendrograma donde las observaciones $i$ e $i'$ se unen por primera vez en un *cluster* común. Dado que las distancias cofenéticas deben cumplir estrictamente la desigualdad ultramétrica ($C_{ii'} \le \max\{C_{ik}, C_{i'k}\}$), es poco común que los datos reales de entrada se acoplen perfectamente a esta restricción métrica. Una baja correlación cofenética advierte al diseñador que el dendrograma está forzando una jerarquía artificial donde los datos reales no la presentan.

### Criterios probabilísticos y de teoría de la información

Cuando se trabaja con modelos de agrupamiento basados en probabilidad (como los modelos de mezclas Gaussianas o GMM entrenados mediante el algoritmo EM), las distancias geométricas puras dejan de ser fiables debido a que los grupos pueden adoptar formas elípticas, orientaciones oblicuas o densidades muy variadas.

#### Criterio de información de Akaike (AIC) y Bayesiano (BIC)

Si evaluamos un modelo probabilístico únicamente por su log-verosimilitud ($LL$), el modelo con mayor número de parámetros o componentes siempre parecerá mejor en los datos de entrenamiento, promoviendo el sobreajuste. Para evitar esto, AIC y BIC añaden penalizaciones matemáticas que balancean el ajuste y la complejidad:

$\text{AIC} = -2LL + 2d$

$\text{BIC} = -2LL + d \log(N)$

Donde $LL$ es la log-verosimilitud máxima del modelo25, $d$ es el número total de parámetros libres a aprender (medias, pesos y covarianzas de los componentes) y $N$ es el tamaño muestral de instancias.

##### Penalización: 

Dado que $\log N > 2$ para cualquier conjunto de datos real ($N > 7$), el BIC penaliza la complejidad de forma mucho más severa que el AIC, priorizando modelos más parsimoniosos (con menor número de componentes).

##### Comportamiento asintótico: 

El BIC es asintóticamente consistente, lo que significa que si el modelo real está dentro de los candidatos, la probabilidad de seleccionarlo tiende a 1 cuando $N \to \infty$. El AIC, por el contrario, tiende a sobreajustar en muestras infinitas, eligiendo modelos demasiado complejos, aunque en muestras pequeñas se comporta de forma más competitiva que el conservador BIC.

#### WAIC (*Widely Applicable Information Criterion*)

AIC y BIC asumen que los parámetros del modelo se aproximan a una distribución Gaussiana y que el modelo no es singular. Sin embargo, en mezclas probabilísticas y modelos de aprendizaje profundo con parámetros correlacionados o redundantes (modelos singulares), estas suposiciones fallan.

El WAIC es una métrica de información bayesiana que estima la densidad predictiva posterior esperada (ELPD) mediante métodos de simulación Monte Carlo, penalizando la complejidad a partir de la varianza posterior de las predicciones de cada punto de datos. Esto proporciona un marco de evaluación robusto que funciona incluso bajo geometrías de parámetros singulares y sobre-parametrizadas.

### Métricas y enfoques de información unificados

#### El principio de mínima longitud de descripción (MDL)

Inspirado en la teoría de la información y la codificación de datos, el principio MDL evalúa el *clustering* bajo la premisa de la compresión de datos. Si un agrupamiento es natural y de alta calidad, debería permitirnos describir o transmitir el conjunto de datos de una forma más compacta (utilizando menos bits de información) que si describiéramos los puntos sin agrupar.

Para transmitir un conjunto de puntos usando MDL, debemos codificar dos elementos:

- La teoría o modelo: la ubicación espacial de los centroides de los *clusters* y sus respectivas asignaciones en bits ($\log_2 K$ bits por muestra).
- Los datos dado el modelo: los valores de las características de cada instancia, expresados únicamente como la desviación residual con respecto a su centroide asignado.

Si la segmentación captura grupos muy densos y reales, la drástica reducción de bits al describir los pequeños residuos individuales compensará con creces el costo extra de transmitir la posición de los centroides, logrando una descripción total minimalista. Si el agrupamiento no es real, el costo total en bits aumentará, señalando que la subdivisión es inútil.

#### Utilidad de categoría (*category utility*)

Utilizada en métodos de agrupamiento incremental y jerárquico como Cobweb o Classit, la utilidad de categoría (CU) mide el incremento en la capacidad de predecir correctamente los atributos de una muestra tras conocer que pertenece a un *cluster* específico, en contraste con la predicción incondicional de los datos3.

Para atributos categóricos o nominales, se formula como:

$\text{CU}(C_1, \dots, C_k) = \frac{\sum_{l=1}^k \Pr[C_l] \sum_i \sum_j \left( \Pr[a_i = v_{ij} \mid C_l]^2 - \Pr[a_i = v_{ij}]^2 \right)}{k}$

Donde $\Pr[a_i = v_{ij} \mid C_l]$ es la probabilidad condicional de que el atributo $i$ tome el valor $v_{ij}$ en el *cluster* $l$, y $\Pr[a_i = v_{ij}]$ es su probabilidad incondicional global.

##### Evitar la fragmentación (división por $k$)

Sin el denominador $k$, la máxima puntuación se obtendría asignando cada instancia individual a su propio *cluster* exclusivo, donde la probabilidad condicional es $1.0$. La división por $k$ actúa como un factor heurístico para castigar la fragmentación excesiva del modelo.

##### Varianza mínima (*acuity*)

Para atributos continuos, la fórmula asume una distribución normal y requiere estimar desviaciones estándar locales y globales. Para evitar que un *cluster*  con un único elemento (o con varianza cero) genere un valor infinito e indeterminado en la ecuación, se impone un parámetro de varianza mínima o *acuity*, que emula el error de medición del sensor.

### Evaluación externa mediante datos etiquetados

Cuando disponemos de etiquetas de referencia externas (*labels* que no se utilizaron para guiar el *clustering*), podemos medir qué tan bien se acoplan las fronteras descubiertas con las clases biológicas o de negocio conocidas.

#### Pureza (*purity*)

Consiste en asignar a cada *cluster* completo la etiqueta de clase más frecuente en su interior y calcular la proporción de elementos clasificados correctamente. Se define formalmente como:

$\text{Purity} = \sum_i \frac{N_i}{N} \max_j (p_{ij})$

Donde $N_i$ es el tamaño del *cluster* $i$, $N$ es el total de datos, y $p_{ij} = N_{ij}/N_i$ es la fracción del *cluster* $i$ ocupada por la clase $j$.

##### Limitación

La pureza no penaliza el número de clusters. Si asignas cada muestra de forma individual a su propio *cluster* ($K=N$), la pureza será de manera trivial igual a $1.0$, invalidando su uso si no se controla de forma independiente el número de clusters.

#### Índice Rand (*Rand Index*, RI) y Rand ajustado (ARI)

El índice Rand analiza la consistencia del agrupamiento en función de parejas de elementos. Examina cada uno de los posibles pares de observaciones del *dataset* y evalúa si la decisión de asignarlos al mismo *cluster* o a *clusters* diferentes fue concordante en la partición estimada ($U$) y en la de referencia ($V$):

$R = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{FP} + \text{FN} + \text{TN}}$

- **TP** (Verdaderos Positivos): número de pares que están en el mismo *cluster* en $U$ y en la misma clase en $V$.
- **TN** (Verdaderos Negativos): número de pares que están en *clusters* separados en $U$ y en diferentes clases en $V$.
- **FP** (Falsos Positivos): número de pares asignados al mismo *cluster* en $U$ pero de diferentes clases en $V$.
- **FN** (Falsos Negativos): número de pares en *clusters* separados en $U$ pero de la misma clase en $V$.
 
El RI oscila entre $0$ (desacuerdo total) y $1$ (concordancia perfecta). Sin embargo, el RI clásico raramente alcanza el valor 0, ya que el azar tiende a inflar su puntuación. El Índice Rand Ajustado (ARI) corrige esto restando la concordancia esperada de dos clasificaciones aleatorias bajo un modelo de distribución hipergeométrica generalizada, asegurando que un agrupamiento al azar obtenga una puntuación promedio de $0.0$.

#### Clases a *clusters* (*Classes-to-Clusters Evaluation*)

Popularizado por suites de software clásico como Weka, este enfoque entrena el modelo de *clustering* ignorando por completo el atributo de clase. Una vez que el espacio se ha segmentado, el evaluador examina la composición de cada *cluster* para asignarle de forma retrospectiva la etiqueta de la clase mayoritaria.

A partir de esta asignación, es posible mapear las fronteras geométricas como si fueran un clasificador supervisado y construir una matriz de confusión clásica que revele las tasas exactas de falsos positivos, falsos negativos y precisión global por clase.

## Ejemplo práctico en Java con SMILE

El repositorio [programacion-avanzada-smile](https://github.com/dmjimenezbravo/programacion-avanzada-smile) incluye el ejemplo completo [`Ejemplo05Clustering.java`](https://github.com/dmjimenezbravo/programacion-avanzada-smile/blob/main/src/main/java/es/usal/smile/Ejemplo05Clustering.java), que aplica los tres algoritmos de esta sección con la librería [SMILE](https://haifengl.github.io/). Se usa el conjunto *Iris* **quitando la columna de la especie**: el algoritmo no ve las etiquetas, pero como sabemos que hay tres especies podemos comparar la partición obtenida con la real mediante un índice externo (ARI).

```java
import java.util.Arrays;
import smile.clustering.CentroidClustering;
import smile.clustering.Clustering;
import smile.clustering.DBSCAN;
import smile.clustering.HierarchicalClustering;
import smile.clustering.KMeans;
import smile.clustering.linkage.WardLinkage;
import smile.data.DataFrame;
import smile.io.Read;
import smile.math.MathEx;
import smile.validation.metric.AdjustedRandIndex;

MathEx.setSeed(42);

DataFrame iris = Read.csv("data/iris.csv", "header=true").factorize("species");
double[][] x = iris.drop("species").toArray();          // sin etiquetas
int[] especieReal = iris.column("species").toIntArray(); // solo para evaluar

// 1. K-means con k = 3: fit(datos, k, maxIteraciones)
CentroidClustering<double[], double[]> kmeans = KMeans.fit(x, 3, 100);
int[] grupos = kmeans.group();
System.out.printf("Distorsion (inercia) = %.3f%n", kmeans.distortion());
System.out.printf("ARI frente a las especies reales = %.4f%n",
        AdjustedRandIndex.of(especieReal, grupos));

// Asignar una muestra nueva al centroide mas cercano
double[] nueva = { 5.9, 3.0, 5.1, 1.8 };
System.out.printf("La muestra cae en el cluster %d%n", kmeans.predict(nueva));

// 2. Metodo del codo: distorsion para distintos valores de k
for (int k = 2; k <= 8; k++) {
    System.out.printf("k = %d  distorsion = %.3f%n", k, KMeans.fit(x, k, 100).distortion());
}

// 3. DBSCAN: fit(datos, minPts, epsilon). No hay que fijar k y detecta ruido
DBSCAN<double[]> dbscan = DBSCAN.fit(x, 5, 0.8);
long ruido = Arrays.stream(dbscan.group()).filter(g -> g == Clustering.OUTLIER).count();
System.out.printf("DBSCAN: %d clusters, %d puntos de ruido%n", dbscan.k(), ruido);

// 4. Clustering jerarquico aglomerativo (enlace de Ward) y corte en 3 grupos
HierarchicalClustering jerarquico = HierarchicalClustering.fit(WardLinkage.of(x));
int[] particion3 = jerarquico.partition(3);
System.out.printf("Jerarquico: ARI = %.4f%n", AdjustedRandIndex.of(especieReal, particion3));
```

Algunas ideas que conviene observar al ejecutarlo:

- `kmeans.distortion()` es la inercia intracluster (WCSS) que minimiza $K$-*means*; el bucle sobre $k$ genera los valores que se representarían en la gráfica del método del codo.
- Los puntos que DBSCAN considera ruido reciben la etiqueta especial `Clustering.OUTLIER` en lugar de un número de *cluster*.
- En el *clustering* jerárquico, primero se construye el criterio de enlace (`WardLinkage`, aunque también existen `SingleLinkage`, `CompleteLinkage` o `UPGMALinkage` para el enlace promedio) y después se corta el dendrograma con `partition(k)`.
- El ejemplo completo compara también *X-means* (elección automática de $k$ mediante BIC) y muestra el efecto de estandarizar las variables antes de agrupar.

Para ejecutarlo desde la raíz del repositorio: `mvn exec:java -Dexec.mainClass=es.usal.smile.Ejemplo05Clustering`.

## Resumen

- **$K$-*means***: simple, rápido, requiere $K$ conocido.
- **Método del codo**: heurística para elegir $K$.
- ***Clustering* jerárquico**: dendrograma, mejor visualización.
- **DBSCAN**: detecta formas arbitrarias y outliers.
- **Silhueta**: métrica para evaluar calidad sin etiquetas.

---

**Siguiente**: En 3.2 aprenderemos a reducir dimensionalidad para simplificar y visualizar datos complejos.
