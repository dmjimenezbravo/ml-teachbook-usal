# Sección 3: aprendizaje no supervisado

Bienvenido a la tercera y última sección principal del curso. Aquí exploraremos técnicas para extraer patrones de datos **sin etiquetas**.

## ¿Qué es aprendizaje no supervisado?

A diferencia de los bloques anteriores enfocados en el aprendizaje supervisado, donde el entrenamiento está guiado por una variable objetivo clara o etiqueta (label), el aprendizaje no supervisado representa la desafiante tarea de enfrentarse a conjuntos de datos que carecen de cualquier tipo de salida esperada o anotación. En este paradigma, únicamente disponemos de una colección de características de entrada $X$ (o variables predictoras), pero no contamos con un "maestro" o supervisor que proporcione las respuestas correctas para guiar el proceso de aprendizaje.

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

## ¿Qué Encontrarás en Esta Sección?

- ***Clustering***: se centrará en los métodos diseñados para encontrar subgrupos homogéneos dentro de las observaciones.
  - Se estudiarán algoritmos clásicos basados en centroides, como $K$-*means*.
  - Estructuras de fusión jerárquica o *clustering* jerárquico.
  - Modelos que definen los grupos según la densidad local del espacio geométrico.
- **Reducción de dimensionalidad**: se explorarán herramientas de proyección lineal para simplificar datos.
  - Técnicas que maximizan la varianza retenida, como PCA. 
  - Algoritmos avanzados de aprendizaje de variedades no lineales, como t-SNE y UMAP.
  - Algoritmos orientados específicamente a la visualización cualitativa y la eliminación de ruido antes de entrenar clasificadores supervisados.

## Nota Importante

El aprendizaje no supervisado es más **arte que ciencia**. No hay una única respuesta "correcta". Tu interpretación del dominio y el *feedback* externo son cruciales.

---

**Empecemos**: dirígete al Capítulo 3.1 para aprender sobre *clustering*, el enfoque no supervisado más popular.
