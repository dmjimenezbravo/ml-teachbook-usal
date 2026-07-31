# Sección 3: Aprendizaje No Supervisado

Bienvenido a la tercera y última sección principal del curso. Aquí exploraremos técnicas para extraer patrones de datos **sin etiquetas**.

## ¿Qué es Aprendizaje No Supervisado?

En aprendizaje no supervisado:
- Tenemos **solo características** (X)
- **No tenemos etiquetas** (sin y)
- Queremos **descubrir patrones ocultos**

Es como explorar un territorio desconocido sin un mapa. La máquina debe encontrar su propia estructura.

## Tres Tipos Principales

### 1. Clustering (Agrupamiento)
Agrupar datos similares:
- Segmentar clientes por comportamiento
- Encontrar comunidades en redes sociales
- Clasificar documentos por tema
- Identificar células cancerosas en imágenes

### 2. Reducción de Dimensionalidad
Simplificar datos complejos:
- Visualizar datos de alta dimensión
- Eliminar ruido
- Acelerar algoritmos posteriores
- Comprensión interpretable

### 3. Detección de Anomalías
Identificar casos raros o inusuales:
- Fraude en transacciones
- Fallos en sistemas
- Comportamiento anómalo de usuarios
- Valores atípicos en datos científicos

## ¿Qué Encontrarás en Esta Sección?

### Capítulo 3.1: Clustering
- **K-Means**: partición de datos en grupos
- **Método del Codo**: seleccionar número óptimo de clusters
- **Clustering Jerárquico**: dendrogramas y fusiones
- **DBSCAN**: agrupamiento basado en densidad

### Capítulo 3.2: Reducción de Dimensionalidad
- **PCA (Principal Component Analysis)**: maximiza varianza explicada
- **t-SNE y UMAP**: visualización no lineal

### Capítulo 3.3: Detección de Anomalías
- **Conceptos de patrones inusuales**: qué es una anomalía
- **Métodos de detección**: aplicaciones prácticas

## Desafío Principal: Evaluación

Sin etiquetas, ¿cómo sabemos si nuestros resultados son correctos?

```
Aprendizaje Supervisado:
Datos → Modelo → Predicción → Comparar con y real → Error

Aprendizaje No Supervisado:
Datos → Modelo → Estructura ??? → ¿Es correcta?
```

Esto es más desafiante y requiere:
- **Intuición del dominio**: ¿tiene sentido el resultado?
- **Métricas internas**: silhueta, índice Davies-Bouldin
- **Validación externa**: etiquetas disponibles después (si las hay)

## Aplicaciones Reales

### E-commerce
Segmentar clientes en grupos para marketing personalizado sin etiquetar manualmente.

### Biología
Agrupar proteínas por similitud estructural para descubrir nuevas familias.

### Astronomía
Detectar anomalías en observaciones de telescopios (candidatos a eventos raros).

### Análisis de Redes
Encontrar comunidades de usuarios con intereses similares en redes sociales.

## Estructura Recomendada

1. **Clustering**: Aprende a agrupar datos
2. **Reducción de Dimensionalidad**: Aprende a simplificar datos complejos
3. **Detección de Anomalías**: Aprende a encontrar patrones raros

## Objetivos de Aprendizaje

Después de completar esta sección, podrás:
- ✓ Implementar K-Means desde cero
- ✓ Elegir el número óptimo de clusters
- ✓ Visualizar datos de alta dimensión
- ✓ Detectar anomalías en datos reales
- ✓ Entender las limitaciones de cada técnica

## Nota Importante

El aprendizaje no supervisado es más **arte que ciencia**. No hay una única respuesta "correcta". Tu interpretación domain and feedback externo son cruciales.

---

**Empecemos**: Dirígete al Capítulo 3.1 para aprender sobre Clustering, el enfoque no supervisado más popular.
