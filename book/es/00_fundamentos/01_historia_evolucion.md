# 1.1 Historia y evolución del aprendizaje automático

## Introducción

Entender la historia del aprendizaje automático nos ayuda a comprender por qué ciertos enfoques funcionan mejor que otros y por qué la investigación actual se enfoca en ciertas direcciones.

Como se ve en la {numref}`fig-historia-evolucion-ml`, el camino recorrido no ha sido lineal, sino una sucesión de avances matemáticos, entusiasmo, inviernos de financiación y resurgimientos.

```{figure} ../../_static/generated/diagrams/es/00_fundamentos_01_historia_evolucion_01.svg
:name: fig-historia-evolucion-ml
:alt: Línea temporal de la historia y evolución del aprendizaje automático, desde los mínimos cuadrados de 1805 hasta la IA Generativa de la década de 2020
:width: 100%
:align: center

Línea temporal de la historia y evolución del aprendizaje automático.
```

## Definición y el cambio de paradigma de programación

La Inteligencia Artificial se define de manera concisa como el esfuerzo por automatizar tareas intelectuales que normalmente realizan los seres humanos. Es un campo vasto que abarca tanto el aprendizaje automático como el aprendizaje profundo, pero que también incluye métodos como la IA simbólica, que dominó el campo desde los años 50 hasta finales de los 80.

El aspecto más revolucionario del aprendizaje automático es que representa un **nuevo paradigma de programación**. En la programación clásica, un humano escribe reglas (algoritmos) que procesan datos para producir respuestas. En cambio, en el aprendizaje automático, el proceso se invierte: la máquina analiza los datos de entrada y las respuestas esperadas para inferir cuáles deberían ser las reglas estadísticamente. Para que este aprendizaje ocurra, se requieren **tres elementos clave**: puntos de datos de entrada, ejemplos de la salida esperada (etiquetas) y una forma de medir si el algoritmo está realizando un buen trabajo para ajustar el sistema (función de pérdida).

## Raíces matemáticas y pioneros (1943-1950)

Aunque el término es moderno, muchos conceptos de aprendizaje estadístico se desarrollaron hace decadas. A principios del siglo XIX, Legendre y Gauss publicaron sobre el **método de mínimos cuadrados**, que es la forma más temprana de regresión lineal. En 1936, Fisher propuso el **análisis discriminante lineal** para predecir valores cualitativos.

En el ámbito computacional, el campo comenzó a gestarse en la década de 1940. En 1943, Warren McCulloch y Walter Pitts presentaron el **primer modelo computacional de neuronas biológicas**. Posteriormente, en 1950, Alan Turing publicó su artículo fundamental "Computing Machinery and Intelligence", donde introdujo el **Test de Turing** como una herramienta conceptual para discutir la naturaleza de la cognición en las máquinas y la posibilidad de que estas emularan la inteligencia humana.

```{figure} ../../_static/external_images/alan_turing_1951.jpg
:name: fig-alan-turing
:alt: Retrato fotográfico de Alan Turing tomado en 1951
:width: 40%
:align: center

Alan Turing en 1951, autor del artículo que introdujo el Test de Turing.
Fuente: fotografía de Elliott & Fry (1951), Computer History Museum — dominio público (Wikimedia Commons, obra publicada antes de 1956).
```

## El nacimiento formal y la era simbólica (1956-1980)

La IA como campo de investigación cristalizó formalmente en 1956, cuando John McCarthy organizó el **seminario de Dartmouth**. McCarthy propuso la conjetura de que cada aspecto del aprendizaje o la inteligencia podía describirse con tal precisión que una máquina podría simularlo. Durante décadas, los expertos creyeron que la inteligencia de nivel humano se alcanzaría mediante la codificación manual de un conjunto masivo de reglas lógicas explícitas. Este enfoque, conocido como **IA simbólica**, alcanzó su auge con los **sistemas expertos** en los años 80.

## Los inviernos de la IA y el resurgimiento del aprendizaje automático

La historia de la IA no ha sido lineal, sino que ha pasado por ciclos de optimismo extremo seguidos de desilusión y recortes de fondos, conocidos como "Inviernos de la IA".

- **Primer Invierno**: ocurrió en los años 70, tras el fracaso de las expectativas de la IA simbólica temprana y de los modelos simples como el **Perceptrón**, que no podían resolver problemas no lineales.
- **Segundo Invierno**: sucedió a principios de los 90, cuando los sistemas expertos resultaron costosos de mantener, difíciles de escalar y limitados en su alcance.

```{figure} ../../_static/external_images/frank_rosenblatt.jpg
:name: fig-rosenblatt
:alt: Retrato fotográfico de Frank Rosenblatt, creador del Perceptrón
:width: 35%
:align: center

Frank Rosenblatt (h. 1950), creador del Perceptrón en 1958.
Fuente: Heinz Nixdorf MuseumsForum — Wikimedia Commons, licencia CC BY-SA 4.0.
```

El Aprendizaje Automático comenzó a florecer realmente en la década de 1990, impulsado por la disponibilidad de hardware más rápido y conjuntos de datos masivos. A diferencia de la IA simbólica, que era apta para problemas lógicos como el ajedrez, el ML permitió abordar problemas "difusos" como el reconocimiento de voz o la clasificación de imágenes.

## Del *Deep Learning* a la era de los LLM (2010-presente)

El **Aprendizaje Profundo** (*Deep Learning*) es una rama especializada del ML que enfatiza el aprendizaje de representaciones sucesivas en capas. Estas capas, estructuradas en redes neuronales, actúan como un proceso de destilación de información en múltiples etapas donde los datos se vuelven cada vez más útiles para una tarea específica. El punto de inflexión para el *Deep Learning* ocurrió entre 2011 y 2015, destacando la victoria del grupo de Hinton en el desafío **ImageNet** de 2012.

Durante esta época, el conjunto de datos **MNIST** (dígitos manuscritos del 0 al 9) se convirtió en el banco de pruebas por excelencia para comparar algoritmos de clasificación, desde los primeros clasificadores estadísticos hasta las redes convolucionales profundas.

```{figure} ../../_static/external_images/mnist_dataset_example.png
:name: fig-mnist
:alt: Cuadrícula de ejemplos del dataset MNIST mostrando distintas variantes manuscritas de los dígitos del 0 al 9
:width: 70%
:align: center

Muestras del dataset MNIST: distintas variantes manuscritas de cada dígito (0-9).
Fuente: Suvanjanprasai (2024) — Wikimedia Commons, licencia CC BY-SA 4.0.
```

A partir de 2017, la **arquitectura Transformer**, que utiliza un mecanismo de atención para procesar secuencias sin capas recurrentes, desató una revolución en el procesamiento de lenguaje natural. Esto permitió el desarrollo de **modelos fundacionales** entrenados mediante **aprendizaje auto-supervisado** en cantidades ingentes de datos de internet. La era actual, marcada por la **IA Generativa** (como ChatGPT y Gemini), ha llevado la tecnología a una escala de miles de millones de parámetros, permitiendo no solo clasificar, sino generar contenido creativo y funcional.

## Resumen

- El aprendizaje automático nació como respuesta a la pregunta: ¿pueden las máquinas aprender?
- Ha pasado por ciclos de hype y desencanto.
- Los avances recientes se deben a más datos, mejor hardware y algoritmos innovadores.
- El aprendizaje profundo y los LLMs son el estado del arte actual.

---

**Siguiente**: en el Capítulo 1.2, aprenderemos los conceptos fundamentales que necesitas para trabajar con cualquier modelo de aprendizaje automático.
