# Fundamentos e introducción al aprendizaje automático

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23159335-blue.svg)](https://doi.org/10.5281/zenodo.23159335)
[![Licencia: CC BY 4.0](https://img.shields.io/badge/Licencia-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Libro web](https://img.shields.io/badge/Libro-web-8c1515.svg)](https://dmjimenezbravo.github.io/ml-teachbook-usal/)

Libro de texto abierto del bloque de **aprendizaje automático** de la asignatura **Programación Avanzada** del Grado en Ingeniería Informática de la **Universidad de Salamanca**. Recorre el camino desde los conceptos estadísticos clásicos hasta los modelos actuales, con ejemplos prácticos en **Java** usando la biblioteca [SMILE](https://haifengl.github.io/).

El libro está disponible en **español** e **inglés**, como web y como PDF, y se acompaña de un juego de diapositivas por capítulo.

## Leer el libro

| | Español | English |
|---|---|---|
| Web | [Abrir el libro](https://dmjimenezbravo.github.io/ml-teachbook-usal/es/intro.html) | [Open the book](https://dmjimenezbravo.github.io/ml-teachbook-usal/en/intro.html) |
| PDF | [FundamentosEIntroduccionAlAprendizajeAutomatico.pdf](https://dmjimenezbravo.github.io/ml-teachbook-usal/_static/FundamentosEIntroduccionAlAprendizajeAutomatico.pdf) | [FundamentalsAndIntroductionToMachineLearning.pdf](https://dmjimenezbravo.github.io/ml-teachbook-usal/_static/FundamentalsAndIntroductionToMachineLearning.pdf) |

## Contenido

| Sección | Capítulo | Temas principales |
|---|---|---|
| **1. Fundamentos** | 1.1 Historia y evolución del aprendizaje automático | Cambio de paradigma, pioneros, era simbólica, inviernos de la IA, *deep learning* y LLM |
| | 1.2 Conceptos fundamentales | Datos, *features* y etiquetas; *overfitting*/*underfitting*; validación cruzada; métricas; *pipeline* de ML |
| **2. Aprendizaje supervisado** | 2.1 Regresión | Regresión lineal, mínimos cuadrados, descenso del gradiente, regresión polinomial y regularización |
| | 2.2 Clasificación | Regresión logística, algoritmos clásicos, métodos de ensamble, introducción a las redes neuronales, evaluación |
| **3. Aprendizaje no supervisado** | 3.1 *Clustering* | Algoritmos de agrupamiento y técnicas de evaluación |
| | 3.2 Reducción de dimensionalidad | Maldición de la dimensionalidad, PCA, *manifold learning* (t-SNE, UMAP) |
| | 3.3 Detección de anomalías | Enfoques estadísticos y de reconstrucción, algoritmos no supervisados |

Los capítulos 2.1, 2.2, 3.1 y 3.2 incluyen un **ejemplo práctico en Java con SMILE**. El libro termina con la bibliografía del curso: libros de referencia y artículos originales citados.

## Diapositivas

La carpeta [`slides/`](slides/) contiene una presentación Beamer por sección y por idioma, con su fuente LaTeX y el PDF compilado:

| Sección | Español | English |
|---|---|---|
| 1. Fundamentos | [`01_fundamentos_es.pdf`](slides/01_fundamentos_es.pdf) | [`01_fundamentos_en.pdf`](slides/01_fundamentos_en.pdf) |
| 2. Aprendizaje supervisado | [`02_aprendizaje_supervisado_es.pdf`](slides/02_aprendizaje_supervisado_es.pdf) | [`02_aprendizaje_supervisado_en.pdf`](slides/02_aprendizaje_supervisado_en.pdf) |
| 3. Aprendizaje no supervisado | [`03_aprendizaje_no_supervisado_es.pdf`](slides/03_aprendizaje_no_supervisado_es.pdf) | [`03_aprendizaje_no_supervisado_en.pdf`](slides/03_aprendizaje_no_supervisado_en.pdf) |

## Cómo citar

> Jiménez-Bravo, D. M. (2026). *Fundamentos e introducción al aprendizaje automático*. Universidad de Salamanca. https://doi.org/10.5281/zenodo.23159335

Los metadatos de cita están en [`CITATION.cff`](CITATION.cff), [`CITATION.bib`](CITATION.bib) y [`.zenodo.json`](.zenodo.json), y en la página [Cómo citar](https://dmjimenezbravo.github.io/ml-teachbook-usal/es/92_como_citar.html) del libro.

## Licencia

- **Libro** (web y PDF): [Creative Commons Reconocimiento 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/deed.es). Puedes compartirlo y adaptarlo, también con fines comerciales, siempre que cites la autoría.
- **Diapositivas**: [Creative Commons Reconocimiento-NoComercial 4.0 (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/deed.es). Igual que la anterior, pero sin uso comercial.

Las figuras y recursos de terceros indican su fuente y licencia en la página donde aparecen.

## Autor

**Diego M. Jiménez-Bravo**, Universidad de Salamanca — [ORCID 0000-0002-0291-7627](https://orcid.org/0000-0002-0291-7627)

---

## Para mantener el libro

### Estructura del repositorio

```
ml-teachbook-usal/
├── book/
│   ├── es/ , en/            # Contenido del libro (Markdown), misma estructura en ambos idiomas
│   ├── _toc_es.yml , _toc_en.yml        # Índice: secciones, capítulos y numeración
│   ├── _config_es.yml , _config_en.yml  # Título, autor, DOI y opciones de web y PDF
│   └── _static/             # Imágenes, figuras generadas, bibliografía (references.bib) y PDF del libro
├── slides/                  # Diapositivas Beamer (.tex y .pdf)
├── diagram_sources/         # Fuentes Mermaid de los diagramas del libro
├── latex_templates/         # Plantilla LaTeX del PDF (portada, estilos, idioma)
├── scripts/                 # Compilación, vista previa, exportación a PDF y comprobaciones
├── tests/                   # Tests unitarios (unittest)
└── .github/workflows/       # Publicación automática en GitHub Pages
```

Cualquier página que se añada en un idioma debe existir también en el otro, y ambos `_toc_*.yml` deben tener la misma estructura.

### Compilar en local

Requiere Python 3.12. Todos los comandos se ejecutan desde la raíz del repositorio con el Python del entorno `.venv` (en Windows, `.venv\Scripts\python.exe`; en macOS/Linux, `.venv/bin/python`).

| Tarea | Comando |
|---|---|
| Crear el entorno | `python scripts/setup_env.py --yes` |
| Vista previa en `localhost:8000` | `python scripts/launch_preview.py --background` |
| Compilar la web | `python scripts/build_book.py` |
| Preparar LaTeX para PDF | `python scripts/setup_env.py --yes --extras pdf` y `python scripts/setup_latex.py --yes --full` |
| Generar los PDF del libro | `python scripts/export_pdf.py --engine auto` |
| Regenerar los diagramas | `python scripts/render_diagrams.py` |
| Comprobar idiomas y codificación | `python scripts/check_multilang_integrity.py` |
| Comprobar imágenes antes de publicar | `python scripts/optimize_static_assets.py --check` (y `--fix` si lo pide) |
| Tests | `python -m unittest discover -s tests` |

Las diapositivas se compilan con Tectonic desde la carpeta `slides/`, porque enlazan las figuras del libro con rutas relativas:

```bash
cd slides
../.venv/Scripts/tectonic.exe -X compile 01_fundamentos_es.tex   # macOS/Linux: ../.venv/bin/tectonic
```

### Publicación

Cada `push` a `main` lanza la Action [`deploy.yml`](.github/workflows/deploy.yml), que comprueba las imágenes, regenera los PDF del libro en los dos idiomas, compila la web y la publica en GitHub Pages. Los cambios se integran siguiendo git flow: rama `feature/*` → `develop` → `main`.

Hay además dos workflows que solo se lanzan a mano desde la pestaña **Actions**: `sftp-deploy-book`, para publicar en un servidor propio, y `Test Clean Setup`, que prueba una instalación limpia en Windows y macOS.

### Créditos

El libro se ha construido a partir de la plantilla [TeachBook USAL](https://github.com/elloza/teachbook_usal_template), de Álvaro Lozano Murciego y André Filipe Sales Mendes, sobre [TeachBooks](https://teachbooks.io/), [Jupyter Book](https://jupyterbook.org/) y [MyST Markdown](https://mystmd.org/). Las instrucciones para asistentes de código con IA están en [`AGENTS.md`](AGENTS.md).
