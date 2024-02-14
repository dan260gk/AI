# Implementación de algoritmos de descenso de gradiente para la optimización de funciones de costo en modelos de aprendizaje automático utilizando PyTorch. 

**Mejoras que proporciona la optimización de funciones de costo:**

1. **Mejora del rendimiento del modelo:** Al minimizar la función de costo, podemos mejorar el rendimiento del modelo, lo que se traduce en predicciones más precisas y resultados más confiables.

2. **Ajuste de parámetros óptimos:** La optimización de la función de costo nos permite encontrar los valores óptimos de los parámetros del modelo, lo que mejora su capacidad para generalizar a nuevos datos y evitar el sobreajuste.

3. **Selección de modelos:** Al comparar y optimizar diferentes modelos utilizando la función de costo, podemos seleccionar el modelo que mejor se ajuste a nuestros datos y necesidades.

4. **Optimización de hiperparámetros:** Podemos ajustar los hiperparámetros del modelo para mejorar su rendimiento mediante la optimización de la función de costo, lo que nos permite ajustar la complejidad del modelo y controlar el riesgo de sobreajuste.

Ejercicio: utiliza PyTorch y el algoritmo de descenso de gradiente para encontrar el mínimo de la función \( f(x) = x^2 + 3x + 2 \) y visualiza la función junto con el punto mínimo en un gráfico.

1. **Definición de la función objetivo (`f(x)`):**
   En este paso, se define la función que va a optimizar. La función objetivo en este caso es \( f(x) = x^2 + 3x + 2 \).

2. **Creación del tensor de PyTorch para representar el parámetro `x`:**
   Con PyTorch se crear un tensor que represente el parámetro \( x \) que queremos optimizar. En este caso, se inicializa \( x \) con un valor inicial de 0.0 y se especifica `requires_grad=True` para indicar que queremos calcular gradientes con respecto a \( x \) durante el proceso de optimización.

3. **Definición del optimizador y la tasa de aprendizaje:**
   Se utiliza el optimizador de descenso de gradiente estocástico (SGD) de PyTorch para optimizar el valor de \( x \). Específicamente, pasa \( x \) como parámetro al optimizador, junto con una tasa de aprendizaje de 0.1. 
   Nota: La tasa de aprendizaje controla qué tan grandes son los pasos que da el optimizador en la dirección opuesta al gradiente.

4. **Proceso de optimización:**
   En este paso, itera 100 veces para optimizar el valor de \( x \) utilizando el algoritmo de descenso de gradiente. En cada iteración, calcula el valor de la función \( f(x) \) con el valor actual de \( x \). Luego, se realiza el backpropagation para calcular el gradiente de \( f(x) \) con respecto a \( x \) utilizando el método `backward()`. Después de eso, se actualiza el valor de \( x \) utilizando el optimizador llamando al método `step()`.

5. **Impresión del valor óptimo de \( x \) y el mínimo de la función:**
   Después de completar el proceso de optimización, se imprime el valor óptimo de \( x \) encontrado por el algoritmo de optimización y el minimo de la función \( f(x) \) correspondiente a ese valor de \( x \).

6. **Graficación de la función y el mínimo:**
   Finalmente, graficamos la función \( f(x) \) en un rango de valores de \( x \) de -4 a 2 utilizando Matplotlib. También graficamos el punto mínimo de la función en el gráfico, que es el resultado de la optimización. El punto mínimo se muestra en rojo en el gráfico.

En el código proporcionado, estamos optimizando la función de costo \( f(x) = x^2 + 3x + 2 \) utilizando el algoritmo de descenso de gradiente. La optimización de esta función nos proporciona el valor óptimo de \( x \) que minimiza la función y el valor mínimo de la función.

Al minimizar esta función de costo, mejoramos nuestra comprensión de cómo cambian los valores de \( x \) afectan a la función \( f(x) \), lo que nos permite hacer predicciones más precisas y tomar decisiones más informadas en contextos donde esta función se aplique, como en la optimización de modelos de aprendizaje automático.

### Fuentes
[Fuente de cosulta](https://www.deeplearningbook.org/contents/numerical.html)


