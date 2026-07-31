# Libro: Fundamentos e introducción al aprendizaje automático

## Sección 1: Fundamentos

### Capítulo 1.1: Historia y evolución del aprendizaje automático

    Propósito Pedagógico: Que el estudiante comprenda el cambio de paradigma de la programación tradicional (reglas fijas) al aprendizaje basado en datos (inferencia de reglas).
    
        Subsección 1.1.1: Origen: El campo nació en la década de 1950 bajo la premisa de si las máquinas podían "pensar". Alan Turing (1950) propuso un test conceptual para discutir la naturaleza de la cognición, mientras que McCulloch-Pitts (1943) sentaron las bases de las neuronas biológicas computacionales.
        Subsección 1.1.2: Conexionismo temprano (1950s-60s): Época del Perceptrón de Rosenblatt y el optimismo inicial sobre la simulación de la inteligencia.
        Subsección 1.1.3: Invierno de la IA (70s-80s): Periodo donde la IA Simbólica (reglas "si-entonces") dominó pero falló en problemas complejos, llevando a recortes de fondos.
        Subsección 1.1.4: Resurgimiento (90s-2000s): El enfoque cambió hacia el Aprendizaje Automático (ML), donde la máquina infiere reglas a partir de datos y respuestas.
        Subsección 1.1.5: Deep Learning (2010+): Surgimiento de redes neuronales profundas que aprenden representaciones en capas sucesivas.
        Subsección 1.1.6: Era LLM (2020+): Dominada por la arquitectura Transformer y el aprendizaje auto-supervisado sobre datos masivos de internet.
    

### Capítulo 1.2: Conceptos fundamentales
    Propósito Pedagógico: Dominar el vocabulario técnico y los protocolos de evaluación para garantizar que los modelos funcionen en el mundo real.

        Subsección 1.2.1: Datos, features y etiquetas: Una muestra es un punto de datos; las features son sus atributos y la etiqueta (label) es el objetivo a predecir.
        Subsección 1.2.2: Overfitting/Underfitting: La tensión central entre optimizar el rendimiento en el entrenamiento y la capacidad de generalización a datos nuevos.
        Subsección 1.2.3: Validación cruzada: Técnica de remuestreo (como K-fold) para evaluar la precisión del modelo sin sesgos.
        Subsección 1.2.4: Métricas: Uso de precisión, recall, F1 y AUC para evaluar clasificadores, y MSE para regresión.
        Subsección 1.2.5: Pipeline de ML: Flujo que incluye recolección de datos, arquitectura, compilación y entrenamiento.
    
## Sección 2: Aprendizaje supervisado

### Capítulo 2.1: Regresión 
    Propósito Pedagógico: Aprender a modelar variables continuas y controlar la complejidad del modelo mediante penalizaciones matemáticas.

        Subsección 2.1.1: Regresión Lineal: Predicción de una respuesta cuantitativa asumiendo una relación lineal entre variables.
        Subsección 2.1.2: Mínimos cuadrados ordinarios (OLS): Método clásico para minimizar la suma de los cuadrados de las diferencias entre el valor real y el predicho.
        Subsección 2.1.3: Gradiente descendente: Motor de optimización que ajusta los pesos del modelo basándose en la señal de la función de pérdida.
        Subsección 2.1.4: Regresión Polinomial y Regularización (Ridge, Lasso): Métodos para manejar relaciones no lineales y penalizar coeficientes excesivos para evitar el overfitting.
    

### Capítulo 2.2: Clasificación 
    Propósito Pedagógico: Capacitar al estudiante para elegir el algoritmo adecuado según la naturaleza (lineal o no lineal) de los datos categóricos.

        Subsección 2.2.1: Fundamentos
            Regresión Logística: Modelo para respuestas cualitativas que utiliza la función sigmoide para emitir probabilidades entre 0 y 1.
            Matriz de confusión: Tabla para visualizar el rendimiento identificando falsos positivos y negativos.
        Subsección 2.2.2: Algoritmos Clásicos
            Árboles de Decisión: Estratificación del espacio de características mediante reglas de división binaria recursiva.
            Random Forest: Mejora de la precisión mediante el uso de múltiples árboles y bagging (Bootstrap Aggregating).
            SVM: Búsqueda de un hiperplano de margen máximo; el kernel trick permite proyecciones no lineales.
        Subsección 2.2.3: Ensemble Methods
            Boosting (XGBoost/LightGBM): Técnica secuencial donde cada modelo intenta corregir los errores de los anteriores.
        Subsección 2.2.4: Redes Neuronales (Introducción)
            Perceptrón y Backpropagation: Fundamentos del aprendizaje en redes mediante la propagación del error hacia atrás para actualizar pesos.
            Activaciones: Uso de ReLU, sigmoid y softmax para introducir no linealidad y manejar múltiples clases.
    

## Sección 3: Aprendizaje no supervisado

### Capítulo 3.1: Clustering
    Propósito Pedagógico: Descubrir estructuras ocultas y subgrupos en datos que carecen de etiquetas previas.
    
        Subsección 3.1.1: K-Means: Partición de datos en K grupos basados en la cercanía al centroide.
        Método del codo (Elbow): Técnica para seleccionar el número óptimo de clusters analizando la varianza explicada.
        Subsección 3.1.2: Clustering Jerárquico: Construcción de dendrogramas para visualizar fusiones de grupos.
        Subsección 3.1.3: DBSCAN: Agrupamiento basado en la densidad, ideal para identificar formas arbitrarias y detectar outliers.
    

### Capítulo 3.2: Reducción de dimensionalidad
    Propósito Pedagógico: Simplificar conjuntos de datos complejos para mejorar la computación y la interpretabilidad visual.
    
        Subsección 3.2.1: PCA: Transformación de variables correlacionadas en un conjunto menor de componentes principales que retienen la mayor varianza.
        Subsección 3.2.2: t-SNE y UMAP: Técnicas de visualización no lineal para proyectar datos de alta dimensión en espacios 2D o 3D legibles por humanos.
    

### Capítulo 3.3: Detección de anomalías 
    Propósito Pedagógico: Implementar sistemas capaces de alertar sobre eventos raros pero críticos en entornos de producción.

        Subsección 3.3.1: Identificación de patrones inusuales que no se ajustan al comportamiento "normal".
        Subsección 3.3.2: Aplicación en detección de fraudes o fallos en líneas de producción [44, 1.8.2].
    

## Logística Adicional

    Propósito Pedagógico General: El curso busca que el alumno transite desde el análisis estadístico clásico hasta la construcción de modelos modernos de aprendizaje profundo, enfocándose en la resolución de problemas reales.
    Orden Específico: Es crítico estudiar el trade-off sesgo-varianza (Bloque 1) antes de profundizar en la regularización de la regresión y el overfitting en redes neuronales.
    Idiomas:
        Español: Para la instrucción y conceptos didácticos.
        Inglés: Para la terminología de industria (Backpropagation, Feature Engineering, Overfitting).
        Además, crea la versión del libro en español e inglés.
