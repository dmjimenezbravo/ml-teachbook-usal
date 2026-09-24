# 2.2 Clasificación

## Introducción

En clasificación, nuestro objetivo es predecir **categorías** (clases) en lugar de valores continuos. Para clasificar una muestra o instancia con una clase específica los algoritmos de clasificación se basan en los atributos de la muestra. Algunos ejemplos típicos incluyen:

- Clasificación de correos electrónicos: *spam* vs. no-*spam*.
- Clasificación médica: enfermo vs. sano.
- Clasificación de especies: gato vs. perro, etc.
- Etc.

## Fundamentos y regresión logística

A diferencia de la regresión lineal, que predice valores continuos, la clasificación comienza estimando la probabilidad de pertenencia a una categoría.

- **Mecánica del modelo**: la regresión logística utiliza la **función sigmoide** (o logística) para transformar una combinación lineal de las características de entrada en un valor entre 0 y 1. La fórmula se define como: 
$\sigma(t) = \frac{1}{1 + e^{-t}}$

La entrada $t$ de esta función se denomina ***logit*** (o *log-odds*), que representa las probabilidades logarítmicas no normalizadas de la clase positiva.
- **Entrenamiento y optimización**: el modelo se entrena minimizando una función de costo llamada **entropía cruzada** (o *log loss*), que penaliza las predicciones que son seguras pero incorrectas. Dado que esta función es convexa, se puede utilizar el descenso de gradiente para encontrar los pesos óptimos de forma iterativa.
- **Evaluación mediante matriz de confusión**: es una tabla de $K×K$ clases que desglosa el rendimiento del modelo comparando las etiquetas reales (filas) con las predichas (columnas). De aquí surgen cuatro valores críticos:
  - **Verdaderos Positivos** (TP, *True Positives*) y **Verdaderos Negativos** (TN, *True Negatives*): aciertos del modelo.
  - **Falsos Positivos** (FP, *False Positives*): error de tipo I o falsa alarma.
  - **Falsos Negativos** (FN, *False Negatives*): error de tipo II u omisión.

La {numref}`fig-sigmoid` muestra la forma característica en "S" de la función sigmoide: valores de entrada muy negativos se aplastan hacia 0 y valores muy positivos hacia 1, con el punto de decisión en 0.5.

```{figure} ../../_static/generated/figures/es/sigmoid_function.png
:name: fig-sigmoid
:alt: Gráfica de la función sigmoide mostrando cómo transforma cualquier valor real en una probabilidad entre 0 y 1
:width: 75%
:align: center

Función sigmoide de la regresión logística.
```

La {numref}`fig-islr-default` muestra por qué es necesaria esta transformación: sobre el conjunto de datos *Default*, un ajuste por regresión lineal (izquierda) predice probabilidades absurdas fuera del rango [0, 1], mientras que la regresión logística (derecha) siempre produce probabilidades válidas.

```{figure} ../../_static/book_figures/islr_fig4_2_default_logistic.png
:name: fig-islr-default
:alt: Dos gráficas comparando la regresión lineal y la regresión logística sobre el conjunto de datos Default, tomadas del libro An Introduction to Statistical Learning
:width: 90%
:align: center

Regresión lineal (izquierda) frente a regresión logística (derecha) sobre el conjunto de datos *Default*.
Fuente: James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*, Figura 4.2. Springer. Libro de libre distribución para uso educativo (statlearning.com).
```

## Algoritmos clásicos de clasificación

Estos métodos estratifican el espacio de características o buscan fronteras de separación geométricas.

### Árboles de decisión

Los árboles particionan el espacio de entrada de forma recursiva en "cajas" o rectángulos de alta dimensión.

- **Construcción**: siguen un enfoque de "divide y vencerás" mediante el algoritmo **CART**, que realiza divisiones binarias buscando maximizar la pureza de los nodos (reduciendo métricas de impureza como el índice Gini o la entropía).
- **Poda** (*pruning*): para evitar el sobreajuste, se utiliza la poda de complejidad de costos, que elimina ramas innecesarias basándose en pruebas estadísticas o conjuntos de validación.

La {numref}`fig-tree-partition` ilustra cómo un árbol divide el espacio de características en regiones rectangulares mediante cortes sucesivos perpendiculares a los ejes.

```{figure} ../../_static/generated/figures/es/decision_tree_partition.png
:name: fig-tree-partition
:alt: Diagrama de dispersión mostrando cómo un árbol de decisión divide el espacio en regiones rectangulares mediante cortes en X e Y
:width: 65%
:align: center

Particiones recursivas de un árbol de decisión sobre dos características.
```

### Bosques aleatorios (*random forests*)

Es un conjunto de árboles de decisión diseñado para reducir la varianza y mejorar la robustez. Existen dos técnicas para conformar el bosque:

- ***Bagging***: utiliza el muestreo con reemplazo (*bootstrap*) para entrenar cada árbol con una versión diferente de los datos.
- **Decorrelación**: para asegurar que los árboles sean diversos, en cada división solo se considera un subconjunto aleatorio de características.

La {numref}`fig-bagging` esquematiza el proceso completo: a partir del *dataset* original se generan varias muestras *bootstrap*, cada una entrena un árbol distinto (usando además un subconjunto aleatorio de *features* en cada división) y las predicciones individuales se combinan mediante votación mayoritaria (clasificación) o promedio (regresión).

```{figure} ../../_static/generated/diagrams/es/01_aprendizaje_supervisado_02_clasificacion_01.svg
:name: fig-bagging
:alt: Diagrama de flujo mostrando cómo el bagging genera muestras bootstrap, entrena un árbol por cada una y combina sus predicciones mediante votación
:width: 90%
:align: center

*Bagging*: cada árbol se entrena con una muestra *bootstrap* distinta y las predicciones se combinan por votación.
```

### Máquinas de vectores de soporte (SVM, *Support Vector Machines*)

Este modelo busca encontrar un **hiperplano** que separe las clases con el **margen máximo** posible.

- **Vectores de soporte**: la frontera de decisión está determinada únicamente por las muestras más cercanas al hiperplano; mover otros puntos no altera el modelo.
- **Margen blando y truco del *kernel***: las SVM manejan datos no separables linealmente mediante variables de holgura (*slack variables*) y el truco del *kernel*, que mapea implícitamente los datos a espacios de mayor dimensión para encontrar separaciones complejas.

La {numref}`fig-svm-margin` muestra el hiperplano óptimo (línea continua), el margen máximo (banda gris) y los vectores de soporte (puntos rodeados en verde), que son los únicos que determinan la posición de la frontera.

```{figure} ../../_static/generated/figures/es/svm_margin.png
:name: fig-svm-margin
:alt: Diagrama de dispersión con dos clases separadas por un hiperplano, mostrando el margen máximo y los vectores de soporte rodeados
:width: 65%
:align: center

SVM: hiperplano de margen máximo y vectores de soporte.
```

## Métodos de ensamble avanzados

Combinan múltiples modelos para obtener una predicción superior a la de cualquier modelo individual.

- ***Voting*** y ***stacking***: el voto mayoritario (duro) o el promedio de probabilidades (suave) combina modelos independientes. El *stacking* entrena un meta-modelo que aprende a combinar las salidas de los modelos base.

La {numref}`fig-voting-stacking` compara ambos enfoques: en *voting*, los modelos se entrenan de forma independiente y sus salidas se combinan con una regla fija (voto o promedio); en *stacking*, las predicciones de los modelos base se convierten en la entrada de un meta-modelo que aprende cómo combinarlas.

```{figure} ../../_static/generated/diagrams/es/01_aprendizaje_supervisado_02_clasificacion_03.svg
:name: fig-voting-stacking
:alt: Diagrama de flujo comparando voting, donde los modelos se combinan mediante voto o promedio, y stacking, donde las predicciones de los modelos base alimentan un meta-modelo
:width: 90%
:align: center

*Voting* frente a *stacking*: regla fija de combinación frente a meta-modelo que aprende a combinar.
```

- ***Boosting***: a diferencia del entrenamiento en paralelo de los bosques aleatorios, el *boosting* entrena modelos de forma secuencial, donde cada nuevo predictor intenta corregir los errores cometidos por sus predecesores. Ejemplos modernos incluyen *XGBoost* y *Gradient Boosting*.

La {numref}`fig-boosting` ilustra el entrenamiento secuencial del *boosting*: cada modelo débil se entrena sobre los datos ponderados según los errores del modelo anterior, dando más peso a las instancias mal clasificadas, y la predicción final combina todos los modelos mediante una suma ponderada.

```{figure} ../../_static/generated/diagrams/es/01_aprendizaje_supervisado_02_clasificacion_02.svg
:name: fig-boosting
:alt: Diagrama de flujo mostrando el entrenamiento secuencial del boosting, donde cada modelo corrige los errores del anterior antes de combinarse en una suma ponderada
:width: 100%
:align: center

*Boosting*: los modelos se entrenan secuencialmente, cada uno corrigiendo los errores del anterior.
```

## Introducción a las redes neuronales

Representan el aprendizaje de representaciones sucesivas a través de capas de filtrado.

- **Arquitectura**: se componen de una capa de entrada, múltiples capas ocultas (densamente conectadas) y una capa de salida.
- **Funciones de activación**: introducen no linealidad para aprender patrones complejos. **ReLU** es el estándar para capas ocultas, mientras que **Softmax** se usa en la capa de salida para clasificaciones de múltiples clases.
- **Retropropagación** (*backpropagation*): es el mecanismo central de aprendizaje; utiliza la regla de la cadena del cálculo para propagar el error desde la salida hacia atrás, ajustando los pesos de la red para reducir la pérdida total.

La {numref}`fig-nn-architecture` esquematiza una red totalmente conectada con una capa de entrada, una capa oculta y una capa de salida: cada conexión representa un peso que se ajusta durante el entrenamiento.

```{figure} ../../_static/generated/figures/es/nn_architecture.png
:name: fig-nn-architecture
:alt: Diagrama de una red neuronal totalmente conectada con capa de entrada, capa oculta y capa de salida
:width: 70%
:align: center

Arquitectura de una red neuronal totalmente conectada.
```

## Evalaución del rendimeinto en clasificación

La evaluación es el proceso fundamental para medir la capacidad de generalización de un modelo, es decir, su aptitud para realizar predicciones precisas sobre datos que nunca ha visto anteriormente. Un modelo que rinde excepcionalmente en los datos de entrenamiento pero falla en datos nuevos está sufriendo de sobreajuste (*overfitting*) y carece de utilidad práctica.

### Protocolos de validación y división de datos

Para medir la generalización de forma fiable, es obligatorio dividir el conjunto de datos disponible en particiones independientes.

- **Entrenamiento** (*train*), **validación** (*validation*) y **prueba** (*test*): en situaciones de abundancia de datos, se reserva un conjunto de entrenamiento para ajustar los pesos del modelo, uno de validación para seleccionar la mejor arquitectura o hiperparámetros, y uno de prueba que se mantiene en una "bóveda" para una evaluación final e imparcial.
- **Validación cruzada de $K$ pliegues** ($K$-*fold Cross-Validation*): es la estrategia recomendada cuando los datos son escasos. Consiste en dividir los datos en $K$ particiones; el modelo se entrena $K$ veces, usando cada vez una partición distinta para validación y las $K−1$ restantes para entrenamiento. El resultado final es el promedio de las puntuaciones obtenidas en los $K$ experimentos, lo que reduce la varianza de la estimación.
- **Estratificación**: en tareas de clasificación, es vital asegurar que cada partición mantenga la proporción original de las clases, especialmente cuando estas están desbalanceadas.

### Métricas de éxito para clasificadores

La elección de la métrica debe alinearse con el objetivo del negocio y la naturaleza de los datos.

#### La matriz de confusión
Es una herramienta bidimensional donde las filas representan las clases reales y las columnas las predichas. Permite identificar cuatro valores esenciales:

- **Verdaderos Positivos** (TP, *True Positives*): instancias positivas correctamente clasificadas.
- **Verdaderos Negativos** (TN, *True Negatives*): instancias negativas correctamente clasificadas.
- **Falsos Positivos** (FP, *False Positives*): error de tipo I o "falsa alarma".
- **Falsos Negativos** (FN, *False Negatives*): error de tipo II u "omisión".

#### Métricas derivadas
- **Exactitud** (*accuracy*): fracción de predicciones correctas sobre el total. Puede ser engañosa en conjuntos de datos sesgados (*skewed datasets*); por ejemplo, si el 90% de las muestras pertenecen a una clase, un modelo que siempre prediga esa clase tendrá un 90% de exactitud sin haber aprendido nada.
- **Precisión** (*precision*): capacidad del modelo para no etiquetar como positiva una muestra que es negativa:
$\frac{TP}{TP+FP}$
- **Sensibilidad** (*recall*): capacidad de encontrar todas las muestras positivas reales:
$\frac{TP}{TP+FN}$
- **Puntuación F1** (F1-*score*): media armónica de precisión y sensibilidad, útil cuando se busca un equilibrio entre ambas.
- **ROC AUC**: el área bajo la curva de la característica operativa del receptor mide la probabilidad de que el modelo clasifique una muestra positiva aleatoria por encima de una negativa. Un valor de 1.0 es perfecto y 0.5 equivale a una clasificación aleatoria.

La {numref}`fig-roc-curve` compara un buen clasificador (curva alejada de la diagonal, AUC alto) frente al comportamiento de un clasificador aleatorio (línea diagonal, AUC = 0.5).

```{figure} ../../_static/generated/figures/es/roc_curve.png
:name: fig-roc-curve
:alt: Curva ROC mostrando la tasa de verdaderos positivos frente a la tasa de falsos positivos, comparada con la diagonal de un clasificador aleatorio
:width: 65%
:align: center

Curva ROC y área bajo la curva (AUC).
```

### Estrategias de entrenamiento y diagnóstico
La evaluación no solo ocurre al final, sino que debe guiar todo el proceso de desarrollo.

- **Superar una línea de base** (*baseline*): antes de iniciar con modelos complejos, se debe establecer una línea de base trivial (como un clasificador aleatorio). Si el modelo no puede superar este umbral de sentido común, es probable que los datos no contengan información suficiente o el enfoque sea erróneo.
- **Monitoreo de curvas de aprendizaje**: graficar la pérdida (*loss*) y la exactitud tanto del entrenamiento como de la validación permite detectar el punto exacto de sobreajuste (cuando la pérdida de entrenamiento baja pero la de validación empieza a subir).
- **Parada temprana** (*early stopping*): estrategia que utiliza un *callback* para interrumpir el entrenamiento automáticamente cuando la métrica de validación deja de mejorar, ahorrando tiempo y evitando el sobreajuste.
- **Análisis de errores**: inspeccionar manualmente las muestras donde el modelo falla ayuda a entender qué patrones específicos está confundiendo el sistema (por ejemplo, confundir un "3" con un "5" en reconocimiento de dígitos).
- **A/B *testing***: tras el despliegue, se recomienda realizar pruebas aleatorizadas para medir el impacto real del modelo en comparación con el proceso anterior.

## Resumen

- **Regresión logística**: extensión de regresión lineal para clasificación.
- **Matriz de confusión**: visualiza el desempeño.
- **Métricas**: elige según lo que importa (precisión o recall).
- **Árboles de decisión**: interpretables pero proclives a overfitting.
- ***Random Forest***: ensemble que reduce overfitting.
- **SVM**: encuentra separación óptima con kernel trick.
- **Ensanbles**: combinan varios algoritmos de clasificación.
- **Redes neuronales**: poderosas pero complejas.

---

**Siguiente**: en la Sección 3 exploraremos aprendizaje no supervisado, donde trabajamos con datos sin etiquetas.
