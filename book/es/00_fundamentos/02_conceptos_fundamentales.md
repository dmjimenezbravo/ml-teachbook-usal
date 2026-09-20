# 1.2 Conceptos fundamentales

## Introducción

Para trabajar efectivamente con modelos de aprendizaje automático, necesitamos dominar cierta terminología y comprensión de conceptos clave. Este capítulo construye el vocabulario que usaremos a lo largo del curso.

## Datos, *features* y etiquetas

El éxito de cualquier sistema de ML depende de cómo se represente la información.

```
+----------+---------+---------+---------+--------+
| Muestra  | Feature | Feature | Feature | Label  |
|          | 1       | 2       | 3       |        |
+----------+---------+---------+---------+--------+
| Ejemplo  | Edad    | Peso    | Altura  | Tipo   |
| 1        | 25      | 70      | 175     | Gato   |
| 2        | 8       | 4       | 90      | Gato   |
| 3        | 35      | 80      | 180     | Perro  |
+----------+---------+---------+---------+--------+
```

- **Muestra** (*sample*): es la unidad básica de información, también denominada punto de datos, instancia o ejemplo. En un conjunto de datos tabular, cada fila representa una muestra individual.
- **Atributos y *features***: las muestras se caracterizan por sus *features* (características o atributos), que son variables cuantitativas o cualitativas que miden diferentes aspectos de la instancia. Técnicamente, un "atributo" es un tipo de dato (ej. "edad"), mientras que una "*feature*" suele referirse al atributo más su valor específico (ej. "edad = 25"). En *deep learning*, todas las entradas se vectorizan para ser procesadas como puntos en un espacio geométrico.
- **Etiqueta** (*label*) o *target*: en el aprendizaje supervisado, cada muestra tiene asociada una respuesta correcta denominada etiqueta u objetivo. El conjunto de etiquetas para todo un *dataset* constituye la **"verdad fundamental"** (*ground-truth*). En problemas de clasificación, el objetivo es predecir una categoría (clase), mientras que en regresión se busca un valor numérico continuo o escalar.

```java
import java.util.List;

// Características: pueden ser números, categorías, etc.
List<String> frutas = List.of("metros_cuadrados", "numero_habitaciones", "ubicacion", "año_construccion");

// Etiqueta: lo que queremos predecir
String label = "precio";
```

## La tensión entre optimización y generalización: *overfitting* y *underfitting*

El problema fundamental del ML es la tensión entre ajustar el modelo a los datos conocidos y su capacidad de actuar sobre datos nuevos.

- **Optimización**: es el proceso de ajustar los parámetros de un modelo (pesos) para obtener el mejor rendimiento posible en los datos de entrenamiento.
- **Generalización**: se refiere a qué tan bien funciona el modelo entrenado con datos que nunca ha visto antes
- *Overfitting* (**sobreajuste**): ocurre cuando un modelo es demasiado complejo en relación con la cantidad y el ruido de los datos. El modelo "memoriza" el ruido o patrones aleatorios del entrenamiento que no existen en los datos reales, lo que provoca una pérdida baja en entrenamiento pero muy alta en validación.
- *Underfitting* (**subajuste**): sucede cuando el modelo es demasiado simple para capturar la estructura subyacente de los datos. En este estado, tanto el error de entrenamiento como el de validación son elevados.
- *Trade-off* **sesgo-varianza**: el error de generalización se divide en:
  - **Sesgo** (*bias*): error por suposiciones erróneas (ej. asumir una relación lineal cuando es cuadrática); un sesgo alto causa *underfitting*.
  - **Varianza**: error por excesiva sensibilidad a pequeñas variaciones en el entrenamiento; una varianza alta causa *overfitting*.

La {numref}`fig-overfitting-underfitting` ilustra los tres escenarios ajustando distintos modelos a los mismos datos: una recta demasiado simple (izquierda), un polinomio de grado moderado que sigue la tendencia real (centro) y un polinomio de grado muy alto que memoriza cada punto, incluido el ruido (derecha).

```{figure} ../../_static/generated/figures/es/overfitting_underfitting.png
:name: fig-overfitting-underfitting
:alt: Comparación de *underfitting*, ajuste óptimo y *overfitting* mediante tres modelos polinomiales ajustados a los mismos datos
:width: 100%
:align: center

*Underfitting* (alto sesgo) vs. ajuste óptimo vs. *overfitting* (alta varianza).
```

## Protocolos de evaluación: validación cruzada

Para medir la generalización de forma fiable, no basta con evaluar el modelo sobre los mismos datos de entrenamiento.

- **División de datos**: la práctica estándar es dividir los datos en tres conjuntos: entrenamiento (para aprender los pesos), validación (para elegir hiperparámetros y evitar el "*leakage*" de información) y prueba (para la evaluación final e imparcial).
- **Validación cruzada** $K$-*fold*: cuando los datos son escasos, la división simple puede ser poco representativa. Este método consiste en dividir los datos en $K$ particiones (normalmente 5 o 10). El modelo se entrena $K$ veces; en cada iteración, se usa una partición distinta para validación y las $K-1$ restantes para entrenamiento. El puntaje final es el promedio de los $K$ resultados obtenidos, lo que reduce la varianza de la evaluación.
- **Estratificación**: en clasificación, es vital que cada "*fold*" mantenga la misma proporción de clases que el dataset original para evitar sesgos, proceso denominado $K$-*fold* estratificado.

La {numref}`fig-kfold-cv` muestra un ejemplo con $K=5$: en cada fila (cada "*fold*"), un bloque distinto de datos actúa como conjunto de validación (en rojo) mientras el resto se usa para entrenar (en azul).

```{figure} ../../_static/generated/figures/es/kfold_cross_validation.png
:name: fig-kfold-cv
:alt: Esquema de validación cruzada 5-*fold* mostrando qué bloque de datos se usa para validación en cada una de las cinco iteraciones
:width: 90%
:align: center

Validación cruzada $K$-*fold* con $K=5$: cada bloque de datos actúa una vez como conjunto de validación.
```

## Métricas de rendimiento

Las métricas permiten cuantificar el éxito del modelo y guiar las decisiones técnicas.

- **Matriz de confusión**: es una herramienta bidimensional utilizada para evaluar de forma detallada el rendimiento de un clasificador. Su estructura consiste en una tabla donde cada fila representa la clase real (*ground-truth*) y cada columna representa la clase predicha por el modelo. En un problema de clasificación binaria (dos clases), la matriz contiene cuatro valores fundamentales que permiten desglosar los aciertos y errores del sistema:
  - **Verdaderos Positivos** (TP, *True Positives*): son las instancias que el modelo clasificó correctamente como positivas.
  - **Verdaderos Negativos** (TN, *True Negatives*): son las instancias que el modelo clasificó correctamente como negativas.
  - **Falsos Positivos** (FP, *False Positives*): ocurre cuando el modelo predice incorrectamente que una instancia es positiva, cuando en realidad es negativa. Este tipo de error se conoce también como error de tipo I o "falsa alarma".
  - **Falsos Negativos** (FN, *False Negatives*): ocurre cuando el modelo predice incorrectamente que una instancia es negativa, cuando en realidad es positiva. Se denomina también error de tipo II o "omisión".

La {numref}`fig-matriz-confusion` resume visualmente estos cuatro valores: la diagonal principal (verde) representa los aciertos del modelo, mientras que la diagonal secundaria (roja) representa sus dos tipos de error.

```{figure} ../../_static/generated/figures/es/confusion_matrix.png
:name: fig-matriz-confusion
:alt: Matriz de confusión 2x2 con las celdas Verdadero Positivo, Falso Negativo, Falso Positivo y Verdadero Negativo
:width: 65%
:align: center

Matriz de confusión: aciertos (verde) y errores de tipo I y II (rojo).
```

- **Precisión** (*accuracy*): fracción de predicciones correctas sobre el total. Puede ser engañosa si las clases están desbalanceadas.
        $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$
- *Precision* (**precisión de clase**): capacidad del modelo para no etiquetar como positiva una muestra que es negativa.
        $\text{Precisión} = \frac{TP}{TP + FP}$
- *Recall* (**exhaustividad/sensibilidad**): capacidad del modelo para encontrar todas las muestras positivas (TP/(TP+FN)).
        $\text{Recall} = \frac{TP}{TP + FN}$
- **F1**-*score*: media armónica entre Precision y Recall; es útil cuando se busca un equilibrio entre ambas, penalizando valores extremos.
        $F1 = 2 \times \frac{\text{Precisión} \times \text{Recall}}{\text{Precisión} + \text{Recall}}$
- **AUC-ROC**: el área bajo la curva de la característica operativa del receptor representa la probabilidad de que el modelo clasifique una muestra positiva aleatoria por encima de una negativa. Un AUC de 1.0 es perfecto, mientras que 0.5 equivale a una clasificación aleatoria.

## *Pipeline* de ML: el flujo de trabajo universal

El desarrollo de un proyecto de Aprendizaje Automático sigue un *blueprint* universal que garantiza que el sistema no solo aprenda de los datos, sino que sea capaz de generalizar sus resultados en entornos de producción. La {numref}`fig-pipeline-ml` resume las 10 etapas y su naturaleza cíclica: el ajuste de hiperparámetros suele requerir volver a entrenar el modelo varias veces antes de la evaluación final.

```{figure} ../../_static/generated/diagrams/es/00_fundamentos_02_conceptos_fundamentales_01.svg
:name: fig-pipeline-ml
:alt: Diagrama de flujo del pipeline de aprendizaje automático con sus 10 etapas, desde la recolección de datos hasta el despliegue, incluyendo el bucle de reentrenamiento
:width: 55%
:align: center

El *pipeline* de *Machine Learning*: 10 etapas desde la recolección de datos hasta el despliegue.
```

A continuación, se definen las etapas del *pipeline* de ML integrando los fundamentos teóricos y técnicos de las fuentes:

1. **Recolección de datos**: esta es la fase más ardua y costosa, donde se debe entender el dominio del problema y los objetivos de negocio. Consiste en obtener muestras representativas del entorno real, lo que a menudo implica la anotación o etiquetado manual por parte de expertos humanos para generar la "verdad fundamental" (*ground-truth*) necesaria en el aprendizaje supervisado.
2. **Exploración y limpieza**: es una mala práctica tratar al dataset como una caja negra; se debe visualizar la distribución de los datos mediante histogramas o mapas para detectar anomalías y valores atípicos (*outliers*). La limpieza incluye el tratamiento de valores faltantes (ya sea eliminando registros o imputando promedios) y la detección de errores de etiquetado para evitar que el ruido degrade el rendimiento del modelo.
3. **Ingeniería de características**: consiste en aplicar el conocimiento humano para realizar transformaciones no aprendidas que faciliten la tarea del algoritmo, como normalizar escalas numéricas o estandarizar textos para borrar diferencias de codificación. Aunque el *Deep Learning* automatiza gran parte de este proceso, una buena ingeniería de características permite resolver problemas con muchos menos datos y recursos computacionales.
4. **División** *train*/*test*: para evaluar la capacidad de generalización, los datos deben dividirse estrictamente en conjuntos de entrenamiento, validación y prueba. Es crítico realizar un barajado aleatorio (*shuffling*) para asegurar representatividad, excepto en series temporales, donde el conjunto de prueba debe ser cronológicamente posterior al de entrenamiento para evitar fugas de información del futuro.
5. **Selección del modelo**: en esta etapa se eligen los prejuicios de arquitectura (*architecture priors*) adecuados para la tarea, como redes convolucionales para imágenes o Transformers para lenguaje natural. El primer objetivo técnico es desarrollar un modelo sencillo que logre superar un baseline de sentido común, demostrando que existe un patrón estadístico explotable en los datos.
6. **Entrenamiento**: el entrenamiento es un proceso iterativo llamado bucle de entrenamiento, donde los pesos del modelo se inicializan aleatoriamente y se ajustan gradualmente. Mediante el paso hacia adelante (*forward pass*), el cálculo de la pérdida y la retropropagación del gradiente, el sistema optimiza sus parámetros internos para minimizar el error en los datos de entrenamiento.
7. **Evaluación**: se utiliza el conjunto de validación para monitorear el rendimiento del modelo durante el entrenamiento mediante métricas como la precisión, *recall* o el AUC. Esta fase permite detectar si el modelo está en un estado de subajuste (*underfitting*) o si ha comenzado a memorizar el ruido de los datos de entrenamiento.
8. **Ajuste de hiperparámetros**: los hiperparámetros son configuraciones externas (como el número de capas o la tasa de aprendizaje) que no se aprenden mediante gradiente descendente, sino mediante búsqueda sistemática. Se emplean técnicas como la validación cruzada $K$-*fold* o la optimización bayesiana para encontrar la configuración que maximice la generalización antes de la evaluación final.
9. **Evaluación final**: una vez seleccionado el mejor modelo y ajustados sus hiperparámetros, se realiza una única evaluación sobre el conjunto de prueba, el cual debe haber permanecido en una "bóveda" sin ser consultado previamente. Si los resultados aquí son significativamente peores que en validación, es señal de que hubo un sobreajuste al proceso de validación o una falta de representatividad en los datos.
10. **Despliegue**: el modelo final se exporta para su uso en producción, ya sea como una API web, en aplicaciones móviles o dispositivos embebidos. El *pipeline* no termina aquí; es imperativo implementar un monitoreo constante, dado que los datos reales suelen degradarse con el tiempo (*concept drift*), requiriendo ciclos de reentrenamiento periódico.

## Resumen

- **Muestra**: un punto individual en el conjunto de datos.
- **Características**: variables de entrada que describen una muestra.
- **Etiqueta**: variable de salida que queremos predecir.
- ***Overfitting***: memorizar datos de entrenamiento, mal en prueba.
- ***Underfitting***: modelo demasiado simple, mal en ambos.
- **Validación cruzada**: técnica para evaluar desempeño sin sesgo.
- **Precisión**: ratio de positivos predichos correctamente.
- ***Recall***: ratio de positivos reales encontrados.
- ***Pipeline* de ML**: se compone de 10 etapas, desde la recolección de datos al despliegue en producción.

---

**Siguiente**: ahora que entiendes los fundamentos, pasaremos a la Sección 2: Aprendizaje Supervisado, donde aprenderás a construir modelos prácticos.
