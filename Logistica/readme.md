# Regresión Logística y Cálculo de Probabilidades con Scikit-learn 
**Regresión Logística:**
La regresión logística es un algoritmo de aprendizaje supervisado utilizado para problemas de clasificación. Aunque su nombre sugiere regresión, en realidad se utiliza para problemas de clasificación binaria, es decir, cuando la variable objetivo tiene dos categorías.

**Cálculo de Probabilidades con Scikit-learn:**
Scikit-learn proporciona una implementación de regresión logística a través de la clase `LogisticRegression` en el módulo `linear_model`. Esta clase utiliza el algoritmo de optimización de descenso de gradiente para ajustar los parámetros del modelo.

**Proceso General:**
1. **Preparación de los Datos:** Se preparan los datos de entrada (características) y la variable objetivo (etiquetas).
2. **División de los Datos:** Se dividen los datos en conjuntos de entrenamiento y prueba utilizando la función `train_test_split` del módulo `model_selection`.
3. **Creación del Modelo:** Se instancia un objeto de la clase `LogisticRegression`.
4. **Entrenamiento del Modelo:** Se ajusta el modelo a los datos de entrenamiento utilizando el método `fit`.
5. **Predicción:** Se hacen predicciones sobre el conjunto de prueba utilizando el método `predict`.
6. **Evaluación del Modelo:** Se evalúa la precisión del modelo utilizando métricas como la precisión (`accuracy_score`) proporcionada por el módulo `metrics`.

**Cálculo de Probabilidades:**
La regresión logística estima la probabilidad de que una observación pertenezca a una clase en particular. Utiliza la función sigmoide para transformar la salida del modelo en valores de probabilidad. La fórmula general es:

\[
p(y = 1 | x) = \frac{1}{1 + e^{-(b_0 + b_1x_1 + b_2x_2 + b_3x_3 + b_4x_4 + b_5x_5)}}
\]

Donde:
- \(b_0, b_1, b_2, b_3, b_4, b_5\): Son los coeficientes del modelo (intercepto y coeficientes de las características).
- \(x_1, x_2, x_3, x_4, x_5\): Son los valores de las características para una observación específica.

**Intercepto (b0):**
- En el contexto de la regresión logística, el intercepto es un parámetro del modelo que representa el valor esperado de la variable dependiente cuando todas las variables independientes son iguales a cero.
- Matemáticamente, el intercepto se representa como \( b_0 \) en la fórmula de la regresión logística.
- En el caso de un modelo de regresión logística, el intercepto se interpreta como el logaritmo del cociente de probabilidad de pertenecer a la clase 1 y la clase 0 cuando todas las características son cero.
- En el contexto práctico, el intercepto ayuda a ajustar la línea base de la función sigmoide, que se utiliza para predecir las probabilidades de pertenecer a una clase específica.

**Coeficientes de Características (b1, b2, ..., bn):**
- Los coeficientes de características, también conocidos como coeficientes de regresión, son los parámetros que multiplican a las variables independientes en el modelo.
- En el caso de la regresión logística, cada característica tiene un coeficiente asociado que representa la contribución de esa característica a la predicción de la probabilidad de pertenecer a una clase específica.
- Matemáticamente, los coeficientes de características se representan como \( b_1, b_2, ..., b_n \) en la fórmula de la regresión logística.
- Los coeficientes pueden ser positivos o negativos, lo que indica la dirección y la magnitud de la influencia de cada característica en la probabilidad de pertenencia a la clase objetivo.
- Los coeficientes más grandes en valor absoluto indican una mayor influencia de la característica correspondiente en la predicción.


**Ejemplo:**
En el código proporcionado, se ejemplifica cómo se calcula la probabilidad de que un video obtenga más de 5 "likes" utilizando los coeficientes del modelo y los valores de características específicos. Esto se realiza mediante la fórmula de probabilidad individual.
