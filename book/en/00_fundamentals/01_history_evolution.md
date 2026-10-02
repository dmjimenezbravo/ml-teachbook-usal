# History and Evolution of Machine Learning

## Introduction

Understanding the history of machine learning helps us understand why certain approaches work better than others and why current research focuses on certain directions.

As shown in {numref}`fig-history-evolution-ml`, the path traveled has not been linear, but rather a succession of mathematical breakthroughs, enthusiasm, funding winters, and resurgences.

```{figure} ../../_static/generated/diagrams/en/00_fundamentals_01_history_evolution_01.svg
:name: fig-history-evolution-ml
:alt: Timeline of the history and evolution of machine learning, from the 1805 least squares method to the Generative AI of the 2020s
:width: 100%
:align: center

Timeline of the history and evolution of machine learning.
```

## Definition and the programming paradigm shift

Artificial Intelligence is concisely defined as the effort to automate intellectual tasks normally performed by humans. It is a vast field that encompasses both machine learning and deep learning, but that also includes methods such as symbolic AI, which dominated the field from the 1950s until the late 1980s.

The most revolutionary aspect of machine learning is that it represents a **new programming paradigm**. In classical programming, a human writes rules (algorithms) that process data to produce answers. In machine learning, the process is reversed: the machine analyzes the input data and the expected answers to statistically infer what the rules should be. For this learning to occur, **three key elements** are required: input data points, examples of the expected output (labels), and a way to measure whether the algorithm is doing a good job of adjusting the system (a loss function).

## Mathematical roots and pioneers (1943-1950)

Although the term is modern, many statistical learning concepts were developed decades earlier. In the early 19th century, Legendre and Gauss published on the **method of least squares** {cite:p}`legendre1805leastsquares,gauss1809leastsquares`, which is the earliest form of linear regression. In 1936, Fisher proposed **linear discriminant analysis** {cite:p}`fisher1936lda` for predicting qualitative values.

In the computational domain, the field began to take shape in the 1940s. In 1943, Warren McCulloch and Walter Pitts presented the **first computational model of biological neurons** {cite:p}`mcculloch1943logicalcalculus`. Later, in 1950, Alan Turing published his foundational paper "Computing Machinery and Intelligence" {cite:p}`turing1950computingmachinery`, where he introduced the **Turing Test** as a conceptual tool to discuss the nature of cognition in machines and the possibility that they could emulate human intelligence.

```{figure} ../../_static/external_images/alan_turing_1951.jpg
:name: fig-alan-turing
:alt: Photographic portrait of Alan Turing taken in 1951
:width: 40%
:align: center

Alan Turing in 1951, author of the paper that introduced the Turing Test.
Source: photograph by Elliott & Fry (1951), Computer History Museum — public domain (Wikimedia Commons, published before 1956).
```

## Formal birth and the symbolic era (1956-1980)

AI as a research field formally crystallized in 1956, when John McCarthy organized the **Dartmouth workshop** {cite:p}`mccarthy1955dartmouth`. McCarthy proposed the conjecture that every aspect of learning or intelligence could be described with such precision that a machine could simulate it. For decades, experts believed that human-level intelligence would be achieved by manually coding a massive set of explicit logical rules. This approach, known as **symbolic AI**, reached its peak with **expert systems** in the 1980s.

## The AI winters and the resurgence of machine learning

The history of AI has not been linear; instead, it has gone through cycles of extreme optimism followed by disillusionment and funding cuts, known as "AI winters."

- **First winter**: occurred in the 1970s, following the failure of early symbolic AI's expectations and of simple models such as the **Perceptron** {cite:p}`rosenblatt1958perceptron`, which could not solve non-linear problems.
- **Second winter**: happened in the early 1990s, when expert systems turned out to be expensive to maintain, difficult to scale, and limited in scope.

```{figure} ../../_static/external_images/frank_rosenblatt.jpg
:name: fig-rosenblatt
:alt: Photographic portrait of Frank Rosenblatt, creator of the Perceptron
:width: 35%
:align: center

Frank Rosenblatt (c. 1950), creator of the Perceptron in 1958.
Source: Heinz Nixdorf MuseumsForum — Wikimedia Commons, CC BY-SA 4.0 license.
```

Machine Learning truly began to flourish in the 1990s, driven by the availability of faster hardware and massive datasets. Unlike symbolic AI, which was well suited to logical problems such as chess, ML made it possible to tackle "fuzzy" problems such as speech recognition or image classification.

## From Deep Learning to the LLM era (2010-present)

**Deep Learning** is a specialized branch of ML that emphasizes learning successive layered representations. These layers, structured into neural networks, act as a multi-stage information distillation process where the data becomes increasingly useful for a specific task. The turning point for Deep Learning occurred between 2011 and 2015, highlighted by Hinton's group winning the 2012 **ImageNet** challenge {cite:p}`krizhevsky2012imagenet`.

During this era, the **MNIST** dataset (handwritten digits 0-9) became the quintessential benchmark for comparing classification algorithms, from early statistical classifiers to deep convolutional networks.

```{figure} ../../_static/external_images/mnist_dataset_example.png
:name: fig-mnist
:alt: Grid of MNIST dataset examples showing different handwritten variants of the digits 0 through 9
:width: 70%
:align: center

Samples from the MNIST dataset: different handwritten variants of each digit (0-9).
Source: Suvanjanprasai (2024) — Wikimedia Commons, CC BY-SA 4.0 license.
```

Starting in 2017, the **Transformer architecture** {cite:p}`vaswani2017attention`, which uses an attention mechanism to process sequences without recurrent layers, unleashed a revolution in natural language processing. This enabled the development of **foundation models** trained through **self-supervised learning** on enormous amounts of internet data. The current era, marked by **Generative AI** (such as ChatGPT and Gemini), has pushed the technology to a scale of billions of parameters, allowing it not only to classify but also to generate creative and functional content.

## Summary

- Machine learning was born as a response to the question: can machines learn?
- It has gone through cycles of hype and disillusionment.
- Recent advances stem from more data, better hardware, and innovative algorithms.
- Deep learning and LLMs represent the current state of the art.

---

**Next**: in Chapter 1.2, we will learn the fundamental concepts you need to work with any machine learning model.
