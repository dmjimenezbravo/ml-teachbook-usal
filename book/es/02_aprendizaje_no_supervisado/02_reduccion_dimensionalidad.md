# 3.2 Reducción de dimensionalidad

## Introducción

En la analítica de datos moderna y el diseño de sistemas de inteligencia artificial, es común enfrentarse a conjuntos de datos caracterizados por cientos o miles de variables predictoras. Sin embargo, trabajar de forma directa en estos espacios multidimensionales presenta graves inconvenientes prácticos y teóricos. La reducción de dimensionalidad aborda de manera sistemática este problema buscando proyectar o comprimir los datos en un espacio de baja dimensión (habitualmente 2D o 3D). Este proceso no solo acelera drásticamente la computación y mitiga el sobreajuste (*overfitting*), sino que es una herramienta indispensable para la visualización de datos (*DataViz*) y la extracción de factores latentes explicables.

## La maldición de la dimensionalidad (*the curse of dimensionality*)

La intuición humana está programada para razonar en tres dimensiones físicas, lo que nos dificulta comprender las propiedades geométricas de los espacios de alta dimensionalidad. Muchas propiedades matemáticas de los algoritmos tradicionales de aprendizaje automático se degradan o colapsan debido a este fenómeno, acuñado formalmente por Richard Bellman como la maldición de la dimensionalidad.  

La {numref}`fig-dimension-progression` muestra la progresión desde un punto hasta un hipercubo de $p$ dimensiones: a medida que crece la dimensión, el espacio se vuelve ultra-disperso y las muestras tienden a situarse en la frontera.

```{figure} ../../_static/generated/diagrams/es/02_aprendizaje_no_supervisado_02_reduccion_dimensionalidad_01.svg
:name: fig-dimension-progression
:alt: Diagrama de la progresión de dimensiones desde un punto (0D), intervalo (1D), cuadrado (2D) y cubo (3D) hasta un hipercubo de p dimensiones, con espacio ultra-disperso y muestras en la frontera
:width: 100%
:align: center

Progresión de dimensiones: del punto al hipercubo.
```

### Pérdida de vecindad y espacio disperso

El principal síntoma de la maldición de la dimensionalidad es que, a medida que aumenta la dimensión $p$, el volumen del espacio crece de forma exponencial con respecto a las características. Esto provoca que cualquier conjunto de datos real, por grande que sea, popule de manera extremadamente dispersa el espacio de entrada. Para ilustrar este hecho:

- Si en una sola dimensión ($p=1$) un rango local del 10% representa una muestra representativa de vecindad, para capturar el mismo volumen equivalente del 10% en un espacio de 10 dimensiones ($p=10$), un algoritmo local (como $K$-NN) debe extender su búsqueda cubriendo el 80% del rango de cada una de las variables.
- De este modo, los puntos del vecindario dejan de ser "locales" y el algoritmo pierde su poder estadístico estimador.
- Para mantener una densidad de muestreo constante al añadir variables predictoras, el tamaño del *dataset* requerido crece exponencialmente ($O(N^p)$). En la práctica, obtener un volumen tal de datos es inviable.

### La anomalía geométrica de los extremos

En espacios de alta dimensionalidad, la geometría se vuelve muy contraintuitiva.

- **Atracción por la frontera**: en un cuadrado unitario de dos dimensiones (1x1), la probabilidad de que una muestra elegida al azar se sitúe cerca del borde (a menos de 0,001 de distancia) es de apenas un 0,4%. Sin embargo, en un hipercubo de 10000 dimensiones, esta probabilidad es superior al 99,9999%. Prácticamente todas las muestras residen de forma natural en los bordes y esquinas exteriores del hipercubo.
- **Uniformidad de distancias**: la distancia promedio entre dos puntos elegidos al azar en un hipercubo crece drásticamente con la dimensionalidad. Las distancias relativas se igualan, haciendo que todas las muestras parezcan casi a la misma distancia de las demás y neutralizando la eficacia de las métricas estándar como la distancia euclídea. Esto incrementa severamente el riesgo de sobreajuste de los clasificadores, ya que el modelo se vuelve inestable ante ligeras variaciones.

La {numref}`fig-curse-dimensionality` cuantifica ambos efectos con simulaciones: a la izquierda, el porcentaje de muestras cerca del borde de un hipercubo crece rápidamente con las dimensiones; a la derecha, el ratio entre la distancia mínima y máxima entre puntos aleatorios tiende a 1, es decir, todos los puntos parecen equidistantes.

```{figure} ../../_static/generated/figures/es/curse_of_dimensionality.png
:name: fig-curse-dimensionality
:alt: Dos gráficas mostrando cómo el porcentaje de muestras cerca del borde y el ratio de distancias mínima sobre máxima evolucionan al aumentar las dimensiones
:width: 100%
:align: center

La maldición de la dimensionalidad, cuantificada mediante simulación.
```

## Métodos de proyección lineal: PCA (*Principal Components Analysis*)

El análisis de componentes principales (PCA), desarrollado originalmente a principios del siglo XX por Pearson y Hotelling {cite:p}`pearson1901pca,hotelling1933pca`, es el algoritmo de reducción de dimensionalidad lineal por excelencia. Su objetivo es proyectar ortogonalmente los datos originales $X \in \mathbb{R}^{D}$ en un subespacio lineal de baja dimensión $Z \in \mathbb{R}^{M}$ (donde $M < D$), minimizando la pérdida de información bajo criterios estadísticos estrictos.

### Perspectiva de máxima varianza

Desde el punto de vista geométrico, la primera componente principal (PC1) se define como la dirección o eje del espacio de características a lo largo de la cual los datos varían más. Al proyectar los datos sobre este eje, se preserva el mayor porcentaje de la dispersión de la información original.

La {numref}`fig-pca-max-variance` compara dos proyecciones de los mismos datos: sobre PC1 (izquierda) los puntos proyectados quedan muy dispersos a lo largo del eje y se conserva el 90% de la varianza; sobre otra dirección (derecha) los puntos se amontonan y solo se retiene el 13%.

```{figure} ../../_static/generated/figures/es/pca_max_variance_projection.png
:name: fig-pca-max-variance
:alt: Dos paneles con los mismos datos proyectados sobre la primera componente principal, que retiene el 90% de la varianza, y sobre otra dirección, que retiene solo el 13%
:width: 100%
:align: center

Proyección sobre la dirección de máxima varianza (PC1) frente a otra dirección.
```

La segunda componente principal (PC2) busca la dirección que explique la mayor cantidad posible de la varianza restante bajo la restricción estricta de ser completamente ortogonal (y, por tanto, incorrelacionada) a la primera componente.

Este procedimiento se repite de manera sucesiva para generar hasta $D$ componentes distintas.

La {numref}`fig-pca-directions` muestra este concepto sobre datos reales correlacionados: la flecha roja (PC1) señala la dirección de máxima varianza y la verde (PC2), ortogonal a la anterior, captura la varianza restante.

```{figure} ../../_static/generated/figures/es/pca_projection.png
:name: fig-pca-directions
:alt: Diagrama de dispersión de datos correlacionados con dos flechas mostrando las direcciones de las componentes principales PC1 y PC2
:width: 65%
:align: center

Componentes principales: PC1 captura la máxima varianza, PC2 es ortogonal.
```

### Derivación matemática y la SVD

Para realizar PCA, es imperativo que los datos originales sean previamente centrados (restando la media aritmética de cada variable) y, usualmente, estandarizados para que tengan una varianza unitaria (evitando que variables con escalas métricas arbitrariamente grandes dominen la optimización).

La matriz de covarianza de los datos se define como:

$S = \frac{1}{N} X X^T$ 

Mediante la Descomposición en valores propios de la matriz de covarianza (o a través de la Descomposición en Valores Singulares - SVD de la matriz de datos original $X$), se extraen las direcciones principales: 

$S = V D^2 V^T$ 

Donde las columnas de la matriz $V$ corresponden a los vectores de carga (*loadings*), que representan las direcciones de las componentes principales. El valor propio $\lambda_m$ correspondiente a cada autovector mide directamente la cantidad de varianza explicada por ese componente específico.

### Compresión, reconstrucción y perspectiva de autoensamblador lineal

PCA puede conceptualizarse matemáticamente como un autoencoder lineal. Consta de dos fases principales:

- **Codificador** (*encoder*): convierte el vector de entrada $x_n$ en una representación reducida $z_n = B^T x_n$, donde $B$ contiene los autovectores con mayores autovalores asociados.
- **Decodificador** (*decoder*): reconstruye la proyección aproximada de vuelta al espacio original: $\hat{x}_n = B z_n$.
 
El error promedio de reconstrucción o distorsión cuadrática minimizado por este procedimiento equivale exactamente a la suma de los valores propios de las componentes que han sido descartadas del análisis.

Un ejemplo real de la utilidad de PCA aparece en el conjunto de datos *NCI60*, donde cada muestra tiene 6830 genes (dimensiones). La {numref}`fig-islr-nci60-pca` proyecta estas muestras sobre sus tres primeras componentes principales: pese a la enorme dimensionalidad original, las líneas celulares del mismo tipo de cáncer (mismo color) tienden a agruparse en este espacio reducido.

```{figure} ../../_static/book_figures/islr_fig10_15_nci60_pca.png
:name: fig-islr-nci60-pca
:alt: Dos diagramas de dispersión mostrando la proyección de las líneas celulares NCI60 sobre las tres primeras componentes principales, tomados del libro An Introduction to Statistical Learning
:width: 85%
:align: center

Proyección de las líneas celulares de cáncer *NCI60* (6830 genes) sobre sus tres primeras componentes principales.
Fuente: James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*, Figura 10.15. Springer. Libro de libre distribución para uso educativo (statlearning.com) {cite:p}`james2013islr`.
```

## Métodos no lineales y aprendizaje de variedades (*manifold learning*)

Aunque PCA es sumamente robusto y rápido, sufre una limitación obvia: asume que el subespacio interesante de los datos es lineal (un hiperplano plano). Cuando las relaciones de los datos son intrínsecamente no lineales, como en el clásico ejemplo del *Swiss roll* (un rollo de papel curvo en 3D), la proyección ortogonal lineal de PCA colapsa los puntos distantes y mezcla artificialmente la información.

### La hipótesis de la variedad (*the manifold hypothesis*)

El aprendizaje de variedades (*manifold learning*) asume la validez de la hipótesis de la variedad: todos los datos naturales del mundo real (imágenes de MNIST, rostros, voces o textos) yacen en una variedad de baja dimensión embebida dentro del espacio original de alta dimensión.

- Una variedad (*manifold*) es una superficie continua y curva que, de manera local, se asemeja a un espacio lineal euclidiano plano.
- Por ejemplo, el espacio de imágenes de dígitos escritos a mano (28x28 píxeles = 784 dimensiones) está restringido por leyes físicas e invarianzas (grosor del trazo, inclinación, continuidad) que reducen enormemente los grados de libertad reales del sistema, confinando las muestras viables a una variedad continua de muy baja dimensión.
- El objetivo del *manifold learning* no paramétrico es aprender coordenadas incrustadas para cada punto, de modo que se represente fielmente la topología interna y la distancia a lo largo de la superficie de la variedad.

La {numref}`fig-swiss-roll` muestra el ejemplo clásico del *Swiss roll*: los datos (izquierda) están enrollados en 3D, pero su estructura real es una superficie 2D que puede "desenrollarse" (derecha) preservando las distancias a lo largo de la variedad.

```{figure} ../../_static/generated/figures/es/swiss_roll_manifold.png
:name: fig-swiss-roll
:alt: Comparación entre los datos del *Swiss roll* en 3D y su versión desenrollada en 2D, coloreados según su posición a lo largo de la variedad
:width: 100%
:align: center

El *Swiss roll*: datos enrollados en 3D (izquierda) y su variedad desenrollada en 2D (derecha).
```

### Algoritmos no lineales clave

#### *Kernel* PCA (kPCA)

Para extender PCA a problemas no lineales, se utiliza el truco del kernel. kPCA mapea de forma implícita los datos de entrada a un espacio de Hilbert de dimensiones ultra-altas (o infinitas) $\Phi(x)$, donde las relaciones no lineales complejas se vuelven linealmente separables o proyectables.

- Utilizando funciones de kernel comunes como el RBF (Gaussiano), se pueden modelar proyecciones complejas.
- **Inconveniente**: a diferencia del PCA estándar, kPCA no define un mapeo de proyección directa e invertible para datos fuera de la muestra (*out-of-sample prediction*), y en ocasiones, *kernels* mal ajustados pueden expandir y distorsionar el espacio en lugar de comprimirlo de forma útil.

La {numref}`fig-kernel-pca-trick` ilustra el truco del *kernel* con un ejemplo clásico: dos clases dispuestas en círculos concéntricos (izquierda) son imposibles de separar con una línea recta en 2D. Al aplicar el mapeo $\phi(x_1,x_2)=(x_1,x_2,x_1^2+x_2^2)$ (derecha), las clases quedan a alturas distintas y un simple plano horizontal las separa perfectamente.

```{figure} ../../_static/generated/figures/es/kernel_pca_trick.png
:name: fig-kernel-pca-trick
:alt: Dos gráficas mostrando dos círculos concéntricos no separables linealmente en 2D y su transformación a un espacio 3D donde un plano los separa
:width: 100%
:align: center

El truco del *kernel*: datos no separables en 2D se vuelven separables al proyectarlos a una dimensión adicional.
```

#### t-SNE (*t-distributed Stochastic Neighbor Embedding*)

Propuesto por Maaten y Hinton (2008) {cite:p}`vandermaaten2008tsne`, t-SNE es la técnica no convexa preferida para la visualización cualitativa de agrupamientos complejos en dos dimensiones.

1. **Espacio de alta dimensión** (SNE original): convierte las distancias euclidianas entre muestras en probabilidades condicionales Gaussianas $p_{j|i}$ que denotan similitud. Los puntos cercanos reciben altas probabilidades de vecindad y los lejanos probabilidades infinitesimales.
2. **El problema del hacinamiento** (*crowding problem*): cuando se proyectan datos de alta dimensión a un espacio plano 2D, el volumen del espacio disponible disminuye de forma exponencial. Las distancias medias crecen tanto que, usando aproximaciones normales, las fuerzas de atracción obligan a todos los puntos distantes a agruparse en un núcleo denso e indistinguible en el centro del gráfico.
3. **La solución de la distribución $t$ de *student***: $t$-SNE resuelve esta limitación utilizando una distribución $t$ de *student* con un grado de libertad (equivalente a una distribución Cauchy) en el espacio de baja dimensión51. Al tener colas mucho más pesadas e invertidas en el denominador de la ecuación de probabilidad latente $q_{ij}$, se eliminan las fuerzas de atracción no deseadas entre *clusters* distantes: 

$q_{ij} = \frac{(1 + \|z_i - z_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|z_k - z_l\|^2)^{-1}}$

El gradiente actúa de manera similar a una ley física de repulsión-atracción (similar a fuerzas de galaxias y estrellas), permitiendo que los *clusters* se organicen y separen de forma óptima en el plano visual.

La {numref}`fig-pca-vs-tsne` compara ambos métodos sobre un ejemplo clásico: dos "lunas" entrelazadas embebidas en un espacio de 30 dimensiones. PCA (izquierda), al ser una proyección lineal, únicamente rota los datos y no logra separar las dos clases, que siguen entrelazadas. $t$-SNE (derecha) reorganiza los puntos de forma no lineal preservando las vecindades locales, consiguiendo dos grupos claramente diferenciados.

```{figure} ../../_static/generated/figures/es/pca_vs_tsne.png
:name: fig-pca-vs-tsne
:alt: Comparación entre PCA y t-SNE sobre datos en forma de dos lunas entrelazadas embebidas en 30 dimensiones; PCA no separa las clases mientras que t-SNE las separa en dos grupos distintos
:width: 100%
:align: center

PCA (proyección lineal) frente a $t$-SNE (proyección no lineal) sobre datos con estructura no lineal.
```

#### UMAP (*Uniform Manifold Approximation and Projection*)

UMAP {cite:p}`mcinnes2018umap` es una de las técnicas de aprendizaje de variedades más potentes de la actualidad. Fundamentada en la geometría riemanniana clásica y la topología algebraica, UMAP asume que el espacio de los datos es localmente conexo y que la variedad sobre la que yacen es uniforme.

A diferencia de $t$-SNE, que se enfoca casi exclusivamente en retener vecindades muy locales (relaciones de corto alcance), UMAP es capaz de preservar tanto la estructura local como la estructura global de los datos.

Es matemáticamente mucho más eficiente, lo que se traduce en una velocidad de ejecución sustancialmente mayor sobre conjuntos de datos masivos con millones de muestras.

La {numref}`fig-umap-digits` muestra una proyección UMAP a 2D del conjunto de datos de dígitos manuscritos (64 dimensiones): cada dígito (0-9, un color por clase) forma un grupo compacto y bien diferenciado, y la posición relativa entre grupos también resulta informativa.

```{figure} ../../_static/external_images/umap_digits_projection.png
:name: fig-umap-digits
:alt: Proyección UMAP en dos dimensiones del conjunto de datos de dígitos manuscritos, con diez grupos de colores bien separados, uno por dígito
:width: 75%
:align: center

Proyección UMAP del conjunto de datos *Digits*.
Fuente: documentación de umap-learn, «How to Use UMAP» (umap-learn.readthedocs.io). Copyright (c) 2017, Leland McInnes. Licencia BSD de 3 cláusulas.
```

## Resumen

- **PCA**: rápido, lineal, interpatable características originales.
- **$t$-SNE**: visualización excelente, no lineal, lento para datos grandes.
- **UMAP**: balance entre PCA y $t$-SNE.
- **Varianza explicada**: métrica para elegir número de componentes.
- **Maldición de la dimensionalidad**: motiva reducción.
- **Aplicaciones**: compresión, visualización, preprocesamiento.

---

**Siguiente**: En 3.3 aprenderemos a detectar anomalías, el último tema en aprendizaje no supervisado.
