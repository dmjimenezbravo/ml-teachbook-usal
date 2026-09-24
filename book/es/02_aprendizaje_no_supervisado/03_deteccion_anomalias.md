# 3.3 Detección de anomalías

## Introducción

La detección de anomalías es una tarea no supervisada de gran relevancia industrial orientada a identificar patrones inusuales o eventos sospechosos dentro de un flujo continuo de datos. Se aplica ampliamente en la detección de fraudes financieros, la prevención de intrusiones en redes corporativas, el control de calidad en manufactura y la limpieza automatizada de conjuntos de datos.

## Filosofía de la detección de anomalías

A nivel conceptual, la detección de anomalías se basa en la premisa de que los comportamientos normales (denominados *inliers*) son altamente frecuentes y comparten patrones de comportamiento consistentes, mientras que las anomalías (*outliers*) son eventos raros cuyas características se desvían de manera significativa de la norma.

### *Anomaly detection vs. novelty detection*

Aunque en la práctica estos términos se utilicen en ocasiones como sinónimos, la literatura especializada establece una distinción metodológica crítica basada en la contaminación del conjunto de datos de entrenamiento:

- **Detección de anomalías** (*anomaly/outlier detection*): el algoritmo se entrena sobre un conjunto de datos que no está depurado y que, por tanto, puede contener un porcentaje desconocido de *outliers* o muestras ruidosas infiltradas de forma natural. El objetivo es tanto identificar estos *outliers* dentro del propio *set* de entrenamiento (limpieza de datos) como clasificar correctamente las nuevas observaciones entrantes.
- **Detección de novedades** (*novelty detection*): el algoritmo se entrena bajo la premisa estricta de que el *dataset* de entrenamiento está completamente limpio y exento de anomalías. El objetivo aquí es definir de forma precisa la frontera del comportamiento "normal" conocido para catalogar cualquier patrón nuevo o desconocido como una "novedad".

### El enfoque no supervisado

La detección de anomalías se aborda fundamentalmente mediante aprendizaje no supervisado debido a que, en situaciones del mundo real, no se conoce de antemano qué tipo de anomalía puede ocurrir. Intentar entrenar un clasificador supervisado tradicional (como una regresión logística) resulta ineficaz debido al severo desbalance de clases y a que las anomalías futuras pueden presentar patrones completamente inéditos que el modelo jamás vio en el entrenamiento.

## Enfoques estadísticos y de reconstrucción

Antes de recurrir a modelos más complejos, existen dos aproximaciones clásicas muy eficientes basadas en la estimación de densidad y en la proyección lineal de los datos.  

[Datos Normales (Inliers)] ──▶ Modelado de Densidad/Proyección ──▶ Umbral Establecido
                                                                           │
  [Anomalía (Baja Densidad/Alto Error)] ───────────────────────────────────▼──▶ Alerta / Outlier

### Modelado de densidad mediante mezclas gaussianas (GMM)

Bajo este enfoque, se asume que las observaciones normales se concentran en regiones del espacio de características que presentan una alta densidad de probabilidad.

- Se entrena un **Modelo de Mezcla Gaussiana** (GMM) para aproximar la función de densidad de probabilidad del *dataset*.
- Para clasificar una nueva muestra, se calcula su **densidad** bajo el modelo estimado. Cualquier instancia ubicada en una región de baja densidad por debajo de un umbral preestablecido se marca como anomalía.
- En entornos reales donde la tasa histórica de fallos es conocida (por ejemplo, un 4% de productos defectuosos en una fábrica), el **umbral** se establece de forma matemática seleccionando el percentil correspondiente (el 4% con menor densidad bajo el modelo).

La {numref}`fig-gaussian-anomaly` ilustra el concepto: los contornos azules representan curvas de igual densidad de probabilidad y los puntos rojos marcados con X son anomalías que caen fuera de las regiones de alta densidad.

```{figure} ../../_static/generated/figures/es/gaussian_anomaly_detection.png
:name: fig-gaussian-anomaly
:alt: Contornos de densidad gaussiana con datos normales agrupados en el centro y tres anomalías marcadas fuera de las curvas de densidad
:width: 70%
:align: center

Detección de anomalías mediante estimación de densidad gaussiana.
```

### Enfoque de reconstrucción utilizando PCA

Esta técnica se basa en el principio de que los componentes principales mayoritarios de un análisis PCA capturan las direcciones de máxima varianza que caracterizan al comportamiento normal del sistema.

- El *dataset* se proyecta a un espacio de baja dimensión utilizando PCA y, posteriormente, se reconstruye de vuelta al espacio original utilizando la matriz inversa.
- Para cada instancia, se calcula el **error de reconstrucción** (la distancia cuadrática entre el vector original $x$ y su reconstrucción $\hat{x}$).
- Dado que las componentes principales no capturan las desviaciones inusuales de los *outliers*, las anomalías experimentarán un error de reconstrucción significativamente mayor que las instancias normales, permitiendo su fácil identificación.

La {numref}`fig-pca-reconstruction` ilustra el proceso: los datos normales (azul) se proyectan sobre la línea de PC1 y se reconstruyen con un error mínimo (líneas finas), mientras que las dos anomalías (rojo) están lejos de la dirección principal de los datos y, al reconstruirse sobre esa misma línea, presentan un error de reconstrucción mucho mayor (flechas largas).

```{figure} ../../_static/generated/figures/es/pca_reconstruction_anomaly.png
:name: fig-pca-reconstruction
:alt: Diagrama de dispersión mostrando datos normales proyectados sobre la primera componente principal con bajo error de reconstrucción, y dos anomalías con un error de reconstrucción mucho mayor
:width: 65%
:align: center

Detección de anomalías mediante el error de reconstrucción de PCA.
```

## Algoritmos específicos basados en no-supervisión

### *Isolation forest* (bosque de aislamiento)

Es uno de los algoritmos más eficientes y escalables para la detección de *outliers*, especialmente diseñado para trabajar en espacios de alta dimensionalidad.

- **Mecánica**: a diferencia de los métodos tradicionales que intentan modelar la densidad o los puntos normales, *isolation forest* busca aislar explícitamente cada observación. Para ello, construye un conjunto de árboles de decisión aleatorios. En cada nodo de un árbol, se selecciona una característica al azar y se elige un umbral de corte aleatorio (entre el mínimo y el máximo de esa variable) para dividir los datos en dos. Este proceso de partición recursiva continúa hasta que cada instancia queda aislada en su propia hoja.
- **Intuición**: dado que las anomalías se encuentran alejadas del grueso de la población de datos normales, requieren en promedio significativamente menos particiones aleatorias para ser aisladas. Por lo tanto, aquellas instancias que presenten una longitud de camino promedio más corta hacia la raíz a lo largo del bosque de árboles son catalogadas inmediatamente como anomalías.
  
  Datos normales (densos)   ────────────────▶ Requieren muchas divisiones para aislarse.
  Anomalías (aisladas/raras) ───────────────▶ Se aíslan rápidamente (pocas ramas).

La {numref}`fig-isolation-forest` compara ambos casos: la anomalía (izquierda) queda aislada con solo 2 divisiones aleatorias, mientras que un punto normal (derecha) necesita muchas más divisiones para separarse del resto.

```{figure} ../../_static/generated/figures/es/isolation_forest_concept.png
:name: fig-isolation-forest
:alt: Dos paneles comparando cómo una anomalía se aísla con pocas divisiones aleatorias frente a un punto normal que requiere muchas más divisiones
:width: 100%
:align: center

*Isolation Forest*: las anomalías se aíslan en menos particiones que los puntos normales.
```

### *Local Outlier Factor* (LOF)

Este algoritmo basa su funcionamiento en el análisis de la densidad local de las muestras utilizando un enfoque de vecinos más cercanos ($K$-NN).

- LOF compara la **densidad local** de una instancia con la densidad de sus vecinos más cercanos.
- Una instancia normal tendrá una densidad local similar a la de su entorno. En cambio, un ***outlier* local** presentará una densidad significativamente menor que la de sus vecinos más cercanos (estará más aislado en relación con la densidad de su vecindario inmediato).
- Este método es extremadamente útil para detectar **anomalías locales** que no destacarían en un análisis global porque sus valores absolutos no son extremos, pero sí resultan inusuales para el contexto específico de su clúster de pertenencia.

La {numref}`fig-lof` muestra un caso típico: el punto marcado en rojo no está lejos de todos los datos en términos absolutos, pero su densidad local es mucho menor que la de sus vecinos inmediatos, lo que lo delata como anomalía local.

```{figure} ../../_static/generated/figures/es/lof_concept.png
:name: fig-lof
:alt: Diagrama de dispersión con una región densa, una región dispersa y un punto marcado como outlier local por tener baja densidad respecto a su vecindario
:width: 70%
:align: center

*Local Outlier Factor*: una anomalía local no destaca en un análisis global.
```

### *One-class* SVM (máquinas de vectores de soporte de una clase)

Este algoritmo está optimizado específicamente para la detección de novedades (*novelty detection*) en escenarios donde se dispone de un conjunto de datos limpio para el entrenamiento.

- **Funcionamiento**: en lugar de buscar un hiperplano que separe dos clases, *one-class* SVM proyecta los datos a un espacio de características de alta dimensión mediante un *kernel* y busca separar las instancias de entrenamiento del origen.
- **Frontera de decisión**: esto equivale geométricamente a encontrar la región o hiperesfera de volumen mínimo que encierra a casi la totalidad de las muestras de entrenamiento. Si una nueva observación cae fuera de esta región delimitada por los vectores de soporte de frontera, es clasificada automáticamente como una anomalía o novedad.

La {numref}`fig-ocsvm` muestra un ejemplo con datos de forma irregular: *one-class* SVM traza una frontera no lineal (morado) que envuelve ajustadamente la región de datos normales (azul), de modo que cualquier observación nueva que caiga fuera de ella (cruces rojas) se clasifica como anomalía.

```{figure} ../../_static/generated/figures/es/one_class_svm_boundary.png
:name: fig-ocsvm
:alt: Diagrama de dispersión mostrando una frontera de decisión no lineal que envuelve los datos normales, con tres nuevas observaciones fuera de la frontera marcadas como anomalías
:width: 65%
:align: center

*One-class* SVM: la frontera envuelve la región normal; lo que queda fuera se clasifica como anomalía.
```

## Resumen

- **Anomalías**: eventos raros, contextuales o colectivos.
- **Métodos simples**: gaussiana para 1D, aplicación de PCA.
- **Métodos avanzados**: *Isolation Forest*, *Local Outlier Factor*, *one-class* SVM.

---

¡Felicidades! Has completado el contenido de "Fundamentos e introducción al aprendizaje automático". Ahora tienes bases sólidas para explorar temas más avanzados como **aprendizaje profundo**, **procesamiento de lenguaje natural**, o **visión por computadora**.