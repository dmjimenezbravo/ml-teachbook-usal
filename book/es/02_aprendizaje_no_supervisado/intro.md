# Sección 3: Aprendizaje no supervisado

Bienvenido a la tercera y última sección principal del curso. Aquí exploraremos técnicas para extraer patrones de datos **sin etiquetas**.

## ¿Qué es aprendizaje no supervisado?

A diferencia de los bloques anteriores enfocados en el aprendizaje supervisado, donde el entrenamiento está guiado por una variable objetivo clara o etiqueta (*label*), el aprendizaje no supervisado representa la desafiante tarea de enfrentarse a conjuntos de datos que carecen de cualquier tipo de salida esperada o anotación. En este paradigma, únicamente disponemos de una colección de características de entrada $X$ (o variables predictoras), pero no contamos con un "maestro" o supervisor que proporcione las respuestas correctas para guiar el proceso de aprendizaje.

Técnicamente, mientras que el aprendizaje supervisado busca aproximar una función de mapeo directo entre entradas y salidas o estimar una **distribución condicional** $p(y|x)$, el aprendizaje no supervisado trabaja "a ciegas" para descubrir la **estructura latente**, las asociaciones estadísticas intrínsecas o la **distribución incondicional** de los propios datos $p(x)$. Como célebremente ilustró el científico de la computación Yann LeCun: "Si la inteligencia fuera un pastel, el aprendizaje no supervisado sería el cuerpo del pastel (el bizcocho), el aprendizaje supervisado la cobertura (el glaseado), y el aprendizaje por refuerzo la cereza". Esta analogía subraya que la inmensa mayoría de la información disponible en el mundo real no viene etiquetada, haciendo que este paradigma sea el pilar fundamental sobre el cual se erige el verdadero entendimiento de los datos.

### Tipos principales de aprendizaje no supervisado

1. **_Clustering_** (agrupamiento): su objetivo es agrupar datos similares.
2. **Reducción de dimensionalidad**: simplifica datos complejos.
3. **Detección de anomalías**: este tipo permite identificar casos raros o inusuales.

## Ejemplos y utilidades prácticas

El aprendizaje no supervisado no busca realizar predicciones cuantitativas o cualitativas puntuales. Su valor reside en extraer conocimiento, simplificar estructuras y revelar patrones ocultos para que sean interpretables por el ser humano o sirvan de preparación para otros modelos:

- **Segmentación de mercado** (*market segmentation*): al analizar grandes volúmenes de datos demográficos y transaccionales de consumidores (como ingresos, ocupación o hábitos de gasto), las empresas pueden agrupar a las personas en nichos homogéneos para dirigir campañas publicitarias específicas o desarrollar productos personalizados sin necesidad de definir estas categorías de antemano.
- **Visualización de datos complejos**: proyectar datasets de altísima dimensionalidad (con cientos de atributos) a representaciones legibles de dos o tres dimensiones, preservando la estructura y las relaciones geométricas para que un analista pueda comprender de forma intuitiva cómo se organizan los datos.
- **Sistemas de recomendación y filtrado colaborativo**: agrupar usuarios con gustos afines o productos con características similares para predecir preferencias implícitas basándose en comportamientos de consumo históricos.
- **Detección de anomalías**: modelar el comportamiento estadístico de los datos "normales" para identificar de manera automática aquellas instancias sospechosas que se desvían significativamente del patrón general (por ejemplo, para detectar fraudes financieros o fallos en líneas de manufactura).

## ¿Qué encontrarás en esta sección?

- ***Clustering***:
  - $K$-*means*: agrupamiento basado en centroides, inicialización ($K$-*means*++) y elección de $K$ con el método del codo.
  - *Clustering* jerárquico: enfoque aglomerativo, dendrogramas y criterios de enlace.
  - DBSCAN: agrupamiento basado en densidad, capaz de detectar formas arbitrarias y ruido.
  - Técnicas de evaluación: coeficiente de silueta, métricas internas, criterios de teoría de la información y evaluación externa con datos etiquetados.

- **Reducción de dimensionalidad**:
  - La maldición de la dimensionalidad: por qué trabajar en espacios de muchas variables es un problema.
  - PCA: proyección lineal que maximiza la varianza retenida, SVD y reconstrucción.
  - Aprendizaje de variedades (*manifold learning*): la hipótesis de la variedad y el ejemplo del *Swiss roll*.
  - Métodos no lineales: *Kernel* PCA, $t$-SNE y UMAP.

- **Detección de anomalías**:
  - *Anomaly detection* frente a *novelty detection*: la diferencia según la contaminación de los datos de entrenamiento.
  - Enfoques estadísticos y de reconstrucción: mezclas gaussianas (GMM) y error de reconstrucción con PCA.
  - *Isolation forest*: aislar las observaciones mediante particiones aleatorias.
  - *Local Outlier Factor* (LOF): detectar anomalías locales según la densidad del vecindario.
  - *One-class* SVM: delimitar la región de comportamiento normal.

## Nota importante

El aprendizaje no supervisado es más **arte que ciencia**. No hay una única respuesta "correcta". Tu interpretación del dominio y el *feedback* externo son cruciales.

---

**Empecemos**: dirígete al Capítulo 3.1 para aprender sobre *clustering*, el enfoque no supervisado más popular.
