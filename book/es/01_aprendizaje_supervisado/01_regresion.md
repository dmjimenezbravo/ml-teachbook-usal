# Regresión

## Introducción

La regresión es el problema de predecir un **valor continuo**. En otras palabras, la regresión predice una respuesta cuantitativa o valor escalar continuo $Y$ a partir de una o más **variables predictoras** $X$.

## Regresión lineal

La regresión lineal es una técnica fundamental que asume una relación aproximadamente lineal entre los predictores y la variable objetivo.

  - **Simple vs. múltiple**: en la **regresión lineal simple** se utiliza un único predictor para modelar $Y \approx \beta_0 + \beta_1X$. La **regresión lineal múltiple** extiende este concepto asignando un coeficiente o peso específico a cada una de las p características disponibles: $Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_pX_p + \epsilon$.
  - **Parámetros del modelo**: los coeficientes $β_0$​ (intercepto o sesgo) y $β_1​...β_p$​ (pesos o pendientes) representan el conocimiento aprendido por el modelo a partir de los datos. Técnicamente, en un espacio de entrada de $p$ dimensiones, este modelo representa un **hiperplano**.

La {numref}`fig-linear-fit` muestra un ejemplo con un único predictor: la recta roja es el ajuste OLS (*Ordinary Least Squares*) y las líneas grises verticales representan los residuos (la distancia entre cada punto y la predicción).

```{figure} ../../_static/generated/figures/es/linear_regression_fit.png
:name: fig-linear-fit
:alt: Diagrama de dispersión con una recta de regresión lineal ajustada y los residuos marcados como líneas verticales entre cada punto y la recta
:width: 80%
:align: center

Regresión lineal simple: recta ajustada y residuos.
```

Un ejemplo clásico de la literatura {cite:p}`james2013islr` es el conjunto de datos *Advertising*, donde se ajusta una recta que predice las ventas (`Sales`) a partir de la inversión en publicidad en TV (`TV`):

```{figure} ../../_static/book_figures/islr_fig3_1_advertising.png
:name: fig-islr-advertising
:alt: Diagrama de dispersión de ventas frente a inversión en TV con una recta de regresión ajustada, tomado del libro An Introduction to Statistical Learning
:width: 75%
:align: center

Ajuste por mínimos cuadrados de `Sales` sobre `TV` en el conjunto de datos *Advertising*.
Fuente: James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*, Figura 3.1. Springer. Libro de libre distribución para uso educativo (statlearning.com).
```

## Mínimos cuadrados ordinarios

El método de mínimos cuadrados ordinarios (OLS, *Ordinary Least Squares*) es el enfoque estándar para entrenar modelos lineales, buscando los parámetros que mejor se ajustan a los datos de entrenamiento.

  - **Residuales y RSS**: un residual $e_i ​= y_i ​− \hat{y}_​{i}$​ es la diferencia entre el valor real y el predicho. El objetivo de OLS es minimizar la **suma de los cuadrados de los residuales** (RSS, *Residual Sum of Squares*): $RSS = \sum_{i=1}^{n} (y_i ​− \hat{y}_{​i})^2$.
  - **Solución analítica** (ecuaciones normales): a diferencia de otros modelos complejos, el modelo lineal tiene una solución única y global que puede calcularse directamente mediante álgebra matricial: $\hat{\beta} = (X^TX)^{-1}X^Ty$.
  - **Interpretación geométrica**: el vector de predicciones $\hat{y}$​ es la **proyección ortogonal** del vector de observaciones y sobre el subespacio lineal definido por las columnas de la matriz de características $X$.

## Descenso del gradiente

Para conjuntos de datos masivos que no caben en memoria o modelos con miles de parámetros, resolver la ecuación normal es computacionalmente costoso. El descenso del gradiente (GD, *Gradient Descent*) es el algoritmo de optimización iterativo que impulsa el aprendizaje automático moderno.

  - **Mecánica de optimización**: el algoritmo inicializa los pesos aleatoriamente y los ajusta gradualmente midiendo el gradiente local de la función de pérdida con respecto a los parámetros. Se mueve en la dirección opuesta al gradiente (hacia "abajo" en la curva de error) hasta alcanzar un mínimo.
  - **Tasa de aprendizaje** ($\eta$): es un hiperparámetro crítico que determina el tamaño del paso en cada iteración. Una tasa demasiado alta puede causar divergencia, mientras que una muy baja hará que el entrenamiento sea excesivamente lento.
  - Variantes:
    - *Batch* GD: utiliza todo el conjunto de entrenamiento para cada paso.
    - *Stochastic* GD (SGD): utiliza una sola muestra aleatoria por paso, lo que lo hace mucho más rápido y capaz de manejar flujos de datos en línea.
    - *Mini-batch* GD: procesa pequeños grupos de datos (*batches*), equilibrando la estabilidad del Batch GD y la eficiencia del SGD.
  - **Momentum**: técnica inspirada en la física que ayuda a evitar mínimos locales y acelera la convergencia al tener en cuenta las actualizaciones de peso anteriores.

Como se ve en la {numref}`fig-gradient-descent`, el algoritmo parte de un punto inicial y va dando pasos sucesivos en la dirección de descenso más pronunciada hasta converger al mínimo de la función de pérdida.

```{figure} ../../_static/generated/figures/es/gradient_descent.png
:name: fig-gradient-descent
:alt: Curva de la función de pérdida con los pasos del descenso de gradiente convergiendo hacia el mínimo
:width: 75%
:align: center

Descenso de gradiente: pasos iterativos hacia el mínimo de la pérdida.
```

## Regresión polinomial y regularización

Cuando los datos presentan curvas, se pueden extender los modelos lineales mediante ingeniería de características.

  - **Regresión polinomial**: se añaden potencias de los predictores originales (ej. $X^2$, $X^3$) como nuevas características. Aunque el modelo es no lineal respecto a las entradas originales, sigue siendo un modelo lineal en sus parámetros, lo que permite usar OLS para entrenarlo.
  - **El riesgo del sobreajuste**: aumentar el grado del polinomio aumenta la flexibilidad del modelo, lo que puede llevarlo a memorizar el ruido de los datos de entrenamiento (*overfitting*) y fallar en la generalización.
  - **Técnicas de regularización**: consisten en añadir una penalización a la función de costo por tener pesos grandes, forzando al modelo a ser más simple.
    - ***Ridge*** (L2) {cite:p}`hoerl1970ridge`: añade una penalización proporcional al cuadrado de los pesos. Encoge los coeficientes hacia cero pero nunca los elimina por completo; funciona muy bien cuando hay muchos predictores correlacionados.
    - ***Lasso*** (L1) {cite:p}`tibshirani1996lasso`: añade una penalización proporcional al valor absoluto de los pesos. Tiene la propiedad única de forzar algunos coeficientes a ser exactamente cero, realizando automáticamente selección de variables.
    - ***Elastic Net***: una combinación de *Ridge* y *Lasso* que utiliza ambos tipos de penalización mediante un ratio de mezcla.

La {numref}`fig-regularization-paths` compara cómo evolucionan los coeficientes de un modelo al aumentar la penalización $\lambda$: Ridge los encoge suavemente sin llegar nunca a cero, mientras que Lasso los lleva a cero de forma exacta, seleccionando variables.

```{figure} ../../_static/generated/figures/es/regularization_paths.png
:name: fig-regularization-paths
:alt: Comparación de las trayectorias de los coeficientes de Ridge y Lasso al aumentar la penalización lambda
:width: 90%
:align: center

Trayectorias de los coeficientes: Ridge (encogimiento suave) vs. Lasso (selección de variables).
```

Sobre datos reales, la {numref}`fig-islr-ridge-credit` muestra cómo evolucionan los coeficientes estandarizados de Ridge para el conjunto de datos *Credit*, tanto en función de $\lambda$ (izquierda) como de la norma relativa del coeficiente (derecha).

```{figure} ../../_static/book_figures/islr_fig6_4_ridge_credit.png
:name: fig-islr-ridge-credit
:alt: Dos gráficas mostrando la evolución de los coeficientes de la regresión Ridge para el conjunto de datos Credit, tomadas del libro An Introduction to Statistical Learning
:width: 90%
:align: center

Coeficientes de regresión Ridge estandarizados para el conjunto de datos *Credit*.
Fuente: James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*, Figura 6.4. Springer. Libro de libre distribución para uso educativo (statlearning.com).
```

## Validación y diagnóstico

Evaluar un modelo de regresión requiere métricas específicas y el análisis de la calidad del ajuste.
  - Métricas de error:
    - **MSE** (*Mean Squared Error*): promedio de los errores al cuadrado; penaliza fuertemente las desviaciones grandes.
    $MSE = \frac{1}{n}\sum_{i=1}^{n} (y_i ​− \hat{y}_​{i})^2$
    - **RMSE** (*Root MSE*): raíz cuadrada del MSE, lo que devuelve el error a las mismas unidades que la variable objetivo.
    $RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i ​− \hat{y}_​{i})^2}$ 
    - **MAE** (*Mean Absolute Error*): promedio de los valores absolutos de los errores; es más robusto ante valores atípicos (*outliers*).
    $MAE = \frac{1}{n}\sum_{i=1}^{n} \left| y_i ​− \hat{y}_{i} \right|$ 
  - **Coeficiente de determinación** (R2): mide la proporción de la varianza en $Y$ que es explicada por el modelo. Un valor de 1 indica un ajuste perfecto, mientras que 0 indica que el modelo no es mejor que predecir siempre la media.
  - **Análisis de residuos**: graficar los residuos versus los valores ajustados permite detectar problemas como la no-linealidad. Si el gráfico muestra un patrón (como una curva en forma de U), indica que el modelo lineal es insuficiente.
  - **Curvas de validación**: las curvas de validación son herramientas gráficas fundamentales para diagnosticar el comportamiento del modelo frente a su complejidad.
    - **Diagnóstico de sesgo y varianza**: al graficar el error de entrenamiento y de validación contra la complejidad del modelo (ej. grado del polinomio), podemos identificar:
      - **Bajo ajuste** (*high bias*): errores altos tanto en entrenamiento como en validación.
      - **Sobreajuste** (*high varianza*): error de entrenamiento muy bajo pero error de validación muy alto; existe una brecha significativa entre ambas curvas.
    - **Regla de la desviación estándar**: en la práctica, se suele elegir el modelo más simple que esté dentro de una desviación estándar del error mínimo en la curva de validación para asegurar la parsimonia.

## Ejemplo práctico en Java con SMILE

Los conceptos de esta sección se pueden poner en práctica con [SMILE](https://haifengl.github.io/) (*Statistical Machine Intelligence and Learning Engine*), una librería de aprendizaje automático para Java. El repositorio [programacion-avanzada-smile](https://github.com/dmjimenezbravo/programacion-avanzada-smile) de la asignatura de Programación Avanzada incluye el ejemplo completo [`Ejemplo04Regresion.java`](https://github.com/dmjimenezbravo/programacion-avanzada-smile/blob/main/src/main/java/es/usal/smile/Ejemplo04Regresion.java), del que se extrae el siguiente fragmento.

El conjunto `viviendas.csv` es sintético: el precio se ha generado con una relación lineal conocida más ruido gaussiano ($precio = 1.8 \cdot superficie + 12 \cdot habitaciones + 9 \cdot banos - 1.1 \cdot antiguedad - 6.5 \cdot distancia\_centro + 55 + \varepsilon$), de modo que se puede comprobar si OLS recupera los coeficientes reales.

```java
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.io.Read;
import smile.regression.LASSO;
import smile.regression.LinearModel;
import smile.regression.OLS;
import smile.regression.RidgeRegression;
import smile.validation.CrossValidation;
import smile.validation.metric.R2;
import smile.validation.metric.RMSE;

DataFrame viviendas = Read.csv("data/viviendas.csv", "header=true");
Formula formula = Formula.lhs("precio");   // precio ~ resto de columnas

// Utiles.split permuta las filas y las reparte en entrenamiento/test
Utiles.Particion particion = Utiles.split(viviendas, 0.75);
DataFrame train = particion.train();
DataFrame test = particion.test();

// 1. Minimos cuadrados ordinarios: solucion exacta en un solo paso
LinearModel ols = OLS.fit(formula, train);
System.out.println(ols);   // coeficientes, p-valores, R2... (como summary() en R)
System.out.printf("Termino independiente: %.3f (real: 55)%n", ols.intercept());

// 2. Evaluacion sobre el conjunto de test
double[] yTest = formula.y(test).toDoubleArray();
double[] prediccion = ols.predict(test);
System.out.printf("RMSE = %.3f%n", RMSE.of(yTest, prediccion));
System.out.printf("R2   = %.4f%n", R2.of(yTest, prediccion));

// 3. Regularizacion: Ridge encoge los coeficientes, LASSO puede anularlos
LinearModel ridge = RidgeRegression.fit(formula, train, 0.1);
LinearModel lasso = LASSO.fit(formula, train, new LASSO.Options(0.5));
System.out.printf("Ridge RMSE = %.3f%n", RMSE.of(yTest, ridge.predict(test)));
System.out.printf("LASSO RMSE = %.3f%n", RMSE.of(yTest, lasso.predict(test)));

// 4. Validacion cruzada de 10 pliegues
var cv = CrossValidation.regression(10, formula, viviendas,
        (f, datos) -> OLS.fit(f, datos));
System.out.printf("OLS RMSE medio = %.3f | R2 medio = %.4f%n",
        cv.avg().rmse(), cv.avg().r2());
```

Algunas ideas que conviene observar al ejecutarlo:

- `Formula.lhs("precio")` declara la variable objetivo; el resto de columnas del `DataFrame` se usan como predictoras.
- `OLS.fit` resuelve la ecuación normal de mínimos cuadrados de forma exacta, y el modelo resultante imprime los coeficientes estimados junto a su error estándar y su $p$-valor.
- El ejemplo completo compara también OLS con *Random Forest* y *Gradient Boosting*: con datos generados por un modelo lineal, OLS gana a los modelos basados en árboles, lo que ilustra que «más complejo» no significa «mejor».

Para ejecutarlo desde la raíz del repositorio: `mvn exec:java -Dexec.mainClass=es.usal.smile.Ejemplo04Regresion`.

## Resumen

- **Regresión Lineal**: modelo simple pero poderoso.
- **OLS**: solución exacta, pero costosa computacionalmente
- **Descenso de gradiente**: método general, funciona para cualquier modelo.
- **Regresión polinomial**: captura relaciones no lineales.
- **Regularización**: previene *overfitting* (*Ridge* para retención, *Lasso* para selección).
- **Validación**: métricas específicas para validar la regresión acompañadas de un estudio del proceo de entrenamiento.

---

**Siguiente**: en el siguiente capítulo, aprenderemos a extender estas ideas a **Clasificación**, donde predecimos categorías en lugar de números continuos.
