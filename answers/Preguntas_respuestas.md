# Respuestas - Parte 1: Clasificación con MLP

## Resultados generales obtenidos

Se entrenó un modelo MLP para clasificación multiclase de lesiones dermatológicas usando el split definitivo del proyecto:

* test fijo excluido durante la comparación de modelos;
* validación cruzada estratificada de 5 folds sobre `trainval`;
* imágenes redimensionadas a 64x64;
* 9 clases;
* métricas principales: accuracy, macro-F1 y balanced accuracy.

Resumen de resultados:

| Experimento                         | Accuracy media | Macro-F1 medio | Balanced accuracy medio |
| ----------------------------------- | -------------: | -------------: | ----------------------: |
| MLP_3_dropout_batchnorm_weightdecay |          0.618 |          0.603 |                   0.618 |
| MLP_2_dropout_batchnorm             |          0.607 |          0.597 |                   0.609 |
| MLP_0_baseline                      |          0.600 |          0.597 |                   0.599 |
| MLP_8_early_stopping                |          0.572 |          0.561 |                   0.577 |
| MLP_5_xavier_initialization         |          0.570 |          0.555 |                   0.570 |
| MLP_7_uniform_initialization        |          0.565 |          0.545 |                   0.564 |
| MLP_4_data_augmentation             |          0.557 |          0.542 |                   0.556 |
| MLP_6_he_initialization             |          0.555 |          0.534 |                   0.561 |
| MLP_1_dropout                       |          0.495 |          0.481 |                   0.498 |

El mejor resultado de validación cruzada fue obtenido por `MLP_3_dropout_batchnorm_weightdecay`, que combinó Dropout, Batch Normalization y Weight Decay. La mejora respecto del baseline fue moderada, lo cual es razonable porque una MLP no aprovecha de manera explícita la estructura espacial local de las imágenes.

---

# 1. Dataset y preprocesamiento

## ¿Por qué es necesario redimensionar las imágenes a un tamaño fijo para una MLP?

Porque una MLP recibe un vector de entrada de dimensión fija. En este caso, cada imagen RGB de 64x64 se aplana como un vector de:

[
64 \times 64 \times 3 = 12288
]

features. Si las imágenes tuvieran tamaños distintos, la primera capa lineal no podría tener un número fijo de pesos.

## ¿Qué ventajas ofrece Albumentations frente a otras librerías de transformación como `torchvision.transforms`?

Albumentations suele ser más flexible y eficiente para data augmentation en imágenes. Permite definir pipelines complejos con transformaciones geométricas, fotométricas y espaciales, y se usa mucho en visión computacional. También facilita aplicar transformaciones de forma consistente a imágenes y máscaras, lo cual es útil en problemas de segmentación.

En este trabajo, para la implementación MLP se usaron transformaciones equivalentes con `torchvision.transforms`, principalmente por integración directa con `PIL` y `torch.utils.data.Dataset`. Conceptualmente, el objetivo fue el mismo: modificar artificialmente las imágenes de entrenamiento para evaluar si eso mejora la generalización.

## ¿Qué hace `A.Normalize()`? ¿Por qué es importante antes de entrenar una red?

`Normalize()` reescala los valores de los canales de la imagen usando una media y un desvío estándar. La forma general es:

[
x_{norm} = \frac{x - \mu}{\sigma}
]

Esto ayuda a que las entradas tengan una escala más estable, lo que mejora la optimización y evita que algunos canales dominen simplemente por tener valores numéricos más grandes.

## ¿Por qué convertimos las imágenes a `ToTensorV2()` al final de la pipeline?

Porque PyTorch entrena usando tensores, no imágenes en formato `PIL` o `numpy`. `ToTensorV2()` convierte la imagen al formato esperado por PyTorch, usualmente con dimensiones:

[
C \times H \times W
]

es decir, canales primero. En este trabajo, cuando se usó `torchvision.transforms`, el equivalente fue `transforms.ToTensor()`.

---

# 2. Arquitectura del modelo

## ¿Por qué usamos una red MLP en lugar de una CNN aquí? ¿Qué limitaciones tiene?

La MLP se usa como modelo base simple para tener una primera referencia de desempeño. Su principal ventaja es que es fácil de implementar e interpretar.

Su limitación central es que aplana la imagen y pierde la estructura espacial. Es decir, píxeles vecinos en la imagen dejan de tener una relación espacial explícita dentro del modelo. Por eso, para imágenes, una CNN suele ser más adecuada.

## ¿Qué hace la capa `Flatten()` al principio de la red?

Convierte una imagen con forma:

[
C \times H \times W
]

en un vector de una sola dimensión. Por ejemplo, una imagen RGB de 64x64 pasa de:

[
3 \times 64 \times 64
]

a:

[
12288
]

valores de entrada para la primera capa lineal.

## ¿Qué función de activación se usó? ¿Por qué no usamos `Sigmoid` o `Tanh`?

Se usó `ReLU`. Es una función estándar en redes profundas porque es simple, rápida y reduce el problema de gradientes muy pequeños que puede aparecer con `Sigmoid` o `Tanh`.

`Sigmoid` y `Tanh` pueden saturarse para valores grandes positivos o negativos. En esas zonas, el gradiente es muy pequeño y el entrenamiento se vuelve más lento o inestable.

## ¿Qué parámetro del modelo deberíamos cambiar si aumentamos el tamaño de entrada de la imagen?

Habría que cambiar la dimensión de entrada de la primera capa lineal. Si se pasa de 64x64 a 128x128, el número de features cambia de:

[
64 \times 64 \times 3 = 12288
]

a:

[
128 \times 128 \times 3 = 49152
]

Por lo tanto, la primera `Linear` debe aceptar ese nuevo tamaño.

---

# 3. Entrenamiento y optimización

## ¿Qué hace `optimizer.zero_grad()`?

Pone en cero los gradientes acumulados de los parámetros antes de calcular el nuevo gradiente. En PyTorch, los gradientes se acumulan por defecto. Si no se llamara a `zero_grad()`, cada actualización usaría gradientes mezclados de batches anteriores.

## ¿Por qué usamos `CrossEntropyLoss()` en este caso?

Porque se trata de un problema de clasificación multiclase. `CrossEntropyLoss()` combina internamente `LogSoftmax` y `Negative Log-Likelihood`, por lo que espera logits crudos como salida del modelo y etiquetas enteras como target.

## ¿Cómo afecta la elección del tamaño de batch (`batch_size`) al entrenamiento?

Un batch más chico genera actualizaciones más ruidosas, pero puede ayudar a generalizar y usa menos memoria. Un batch más grande estabiliza el gradiente, pero puede requerir más memoria y a veces generalizar peor.

En este trabajo se usó `batch_size = 32`, que es una elección estándar y razonable para el tamaño del dataset y el modelo MLP.

## ¿Qué pasaría si no usamos `model.eval()` durante la validación?

El modelo seguiría en modo entrenamiento. Eso afecta especialmente a capas como Dropout y BatchNorm. Dropout seguiría apagando neuronas aleatoriamente y BatchNorm seguiría actualizando estadísticas internas. Como consecuencia, la validación no sería consistente ni representaría correctamente el desempeño real del modelo.

---

# 4. Validación y evaluación

## ¿Qué significa una accuracy del 70% en validación pero 90% en entrenamiento?

Indica una brecha entre entrenamiento y validación. El modelo aprende muy bien el conjunto de entrenamiento, pero no generaliza igual de bien a datos no vistos. Eso es un signo típico de overfitting.

## ¿Qué otras métricas podrían ser más relevantes que accuracy en un problema real?

En clasificación médica pueden ser más importantes:

* recall o sensibilidad;
* precision;
* F1-score;
* macro-F1;
* balanced accuracy;
* matriz de confusión por clase.

En este trabajo se reportaron `accuracy`, `macro_f1` y `balanced_accuracy`. Macro-F1 y balanced accuracy son útiles porque tratan a las clases de forma más equilibrada.

## ¿Qué información útil nos da una matriz de confusión que no nos da la accuracy?

La matriz de confusión permite ver qué clases se confunden entre sí. Dos modelos pueden tener accuracy similar, pero cometer errores muy distintos. En un problema dermatológico, no es lo mismo confundir dos lesiones benignas que confundir una lesión maligna con una benigna.

## En el reporte de clasificación, ¿qué representan `precision`, `recall` y `f1-score`?

`Precision` mide, para una clase, qué proporción de las predicciones hechas como esa clase fueron correctas.

`Recall` mide qué proporción de los ejemplos reales de esa clase fueron detectados correctamente.

`F1-score` es la media armónica entre precision y recall. Es útil cuando se busca un balance entre ambos.

---

# 5. TensorBoard y logging

## ¿Qué ventajas tiene usar TensorBoard durante el entrenamiento?

TensorBoard permite visualizar la evolución del entrenamiento: loss, accuracy, macro-F1, histogramas de pesos y comparación entre experimentos. Esto ayuda a detectar overfitting, inestabilidad, convergencia lenta o diferencias entre configuraciones.

## ¿Qué diferencias hay entre loguear `add_scalar`, `add_image` y `add_text`?

`add_scalar` guarda valores numéricos por epoch, como loss o accuracy.

`add_image` guarda imágenes, por ejemplo ejemplos de entrada, predicciones o imágenes mal clasificadas.

`add_text` guarda texto, por ejemplo la configuración de un experimento o comentarios descriptivos.

En este trabajo se usó logging de scalars y texto de configuración. También se guardaron histogramas de pesos con `add_histogram`.

## ¿Por qué es útil guardar visualmente las imágenes de validación en TensorBoard?

Porque permite inspeccionar cualitativamente qué está viendo el modelo. En clasificación de imágenes médicas, esto puede ayudar a detectar errores de preprocesamiento, imágenes mal etiquetadas, problemas de resolución o transformaciones excesivas.

## ¿Cómo se puede comparar el desempeño de distintos experimentos en TensorBoard?

Se puede comparar observando las curvas de loss y métricas para distintos runs. Por ejemplo, se pueden comparar `MLP_0_baseline`, `MLP_1_dropout`, `MLP_2_dropout_batchnorm` y `MLP_3_dropout_batchnorm_weightdecay` para ver qué configuración converge mejor y cuál logra mejor validación.

---

# 6. Generalización y transferencia

## ¿Qué cambios habría que hacer si quisiéramos aplicar este mismo modelo a un dataset con 100 clases?

Habría que cambiar la última capa lineal para que tenga 100 salidas. También habría que actualizar el mapeo `class_to_idx`, el cálculo de métricas y el reporte de clasificación.

Además, probablemente habría que usar un modelo más potente, porque 100 clases implican una tarea más compleja.

## ¿Por qué una CNN suele ser más adecuada que una MLP para clasificación de imágenes?

Porque una CNN conserva y explota la estructura espacial de la imagen. Usa filtros convolucionales que detectan patrones locales como bordes, texturas y formas. Una MLP, en cambio, aplana la imagen y trata cada píxel como una feature independiente.

## ¿Qué problema podríamos tener si entrenamos este modelo con muy pocas imágenes por clase?

El modelo puede sobreajustar fácilmente. Puede memorizar ejemplos de entrenamiento en vez de aprender patrones generales. Esto se vuelve especialmente problemático en imágenes médicas, donde puede haber alta variabilidad visual dentro de una misma clase.

## ¿Cómo podríamos adaptar este pipeline para imágenes en escala de grises?

Habría que cargar las imágenes con un solo canal y ajustar la dimensión de entrada. Para una imagen grayscale de 64x64, la entrada sería:

[
64 \times 64 \times 1 = 4096
]

También habría que cambiar la normalización para usar una sola media y un solo desvío estándar.

---

# 7. Regularización

## Preguntas teóricas

### ¿Qué es la regularización en el contexto del entrenamiento de redes neuronales?

La regularización es un conjunto de técnicas que buscan reducir el overfitting. Su objetivo es que el modelo no memorice el entrenamiento, sino que aprenda patrones que generalicen a datos no vistos.

### ¿Cuál es la diferencia entre `Dropout` y regularización `L2` (`weight_decay`)?

Dropout apaga aleatoriamente neuronas durante el entrenamiento. Esto obliga a la red a no depender demasiado de neuronas específicas.

Weight Decay penaliza pesos grandes agregando una penalización L2 al optimizador. Esto favorece modelos con pesos más pequeños y suaves.

### ¿Qué es `BatchNorm` y cómo ayuda a estabilizar el entrenamiento?

BatchNorm normaliza las activaciones intermedias de la red. Esto puede estabilizar la distribución de entradas que recibe cada capa durante el entrenamiento, haciendo que la optimización sea más estable.

### ¿Cómo se relaciona `BatchNorm` con la velocidad de convergencia?

BatchNorm puede acelerar la convergencia porque estabiliza las activaciones internas. En la práctica, muchas veces permite entrenar con learning rates algo mayores y reduce oscilaciones en la loss.

### ¿Puede `BatchNorm` actuar como regularizador? ¿Por qué?

Sí, puede tener un efecto regularizador leve porque usa estadísticas de batch durante entrenamiento. Esto introduce cierto ruido en las activaciones, lo cual puede mejorar la generalización.

### ¿Qué efectos visuales podrías observar en TensorBoard si hay overfitting?

Se observaría que la loss de entrenamiento sigue bajando mientras la loss de validación se estanca o aumenta. También podría verse que la accuracy de entrenamiento sube mucho más que la accuracy de validación.

### ¿Cómo ayuda la regularización a mejorar la generalización del modelo?

Reduce la capacidad efectiva del modelo de memorizar el conjunto de entrenamiento. Esto puede hacer que el modelo aprenda patrones más robustos y transferibles a datos nuevos.

---

## Actividades de modificación

### 1. Agregar Dropout

Se evaluó `MLP_1_dropout`, agregando `Dropout(p=0.5)` entre capas lineales y activaciones.

Resultado:

* `MLP_0_baseline`: accuracy media = 0.600
* `MLP_1_dropout`: accuracy media = 0.495

En este caso, Dropout solo empeoró el resultado. Probablemente `p=0.5` fue demasiado agresivo para esta arquitectura y este tamaño de dataset.

### 2. Agregar Batch Normalization

Se evaluó `MLP_2_dropout_batchnorm`, agregando `BatchNorm1d` después de capas lineales y antes de `ReLU`.

Resultado:

* `MLP_1_dropout`: accuracy media = 0.495
* `MLP_2_dropout_batchnorm`: accuracy media = 0.607

BatchNorm compensó fuertemente la caída producida por Dropout y mejoró incluso levemente respecto del baseline.

### 3. Aplicar Weight Decay

Se evaluó `MLP_3_dropout_batchnorm_weightdecay`, usando:

```python
weight_decay = 1e-4
```

Resultado:

* `MLP_2_dropout_batchnorm`: accuracy media = 0.607
* `MLP_3_dropout_batchnorm_weightdecay`: accuracy media = 0.618

Esta fue la mejor configuración MLP. La combinación de BatchNorm y Weight Decay produjo la mejor validación media.

### 4. Reducir overfitting con data augmentation

Se evaluó `MLP_4_data_augmentation`, agregando transformaciones suaves durante entrenamiento.

Resultado:

* `MLP_3_dropout_batchnorm_weightdecay`: accuracy media = 0.618
* `MLP_4_data_augmentation`: accuracy media = 0.557

En este caso, la data augmentation no mejoró el MLP. Una posible explicación es que el MLP no modela bien invariancias espaciales, porque trabaja sobre la imagen aplanada. Por eso, transformaciones geométricas pueden hacer la tarea más difícil en vez de mejorarla.

### 5. Early Stopping

Se evaluó `MLP_8_early_stopping`.

Resultado:

* `MLP_8_early_stopping`: accuracy media = 0.572

No superó a la mejor configuración. Aun así, queda implementado como mecanismo para evitar entrenamiento innecesario cuando la validación deja de mejorar.

---

## Preguntas prácticas

### ¿Qué efecto tuvo `BatchNorm` en la estabilidad y velocidad del entrenamiento?

BatchNorm mejoró la performance cuando se combinó con Dropout. El modelo `MLP_2_dropout_batchnorm` superó claramente a `MLP_1_dropout`, indicando que BatchNorm estabilizó el entrenamiento.

### ¿Cambió la performance de validación al combinar `BatchNorm` con `Dropout`?

Sí. Dropout solo obtuvo 0.495 de accuracy media, mientras que Dropout + BatchNorm obtuvo 0.607. La combinación fue mucho mejor que usar Dropout aislado.

### ¿Qué combinación de regularizadores dio mejores resultados en tus pruebas?

La mejor combinación fue:

* Dropout;
* BatchNorm;
* Weight Decay;
* sin data augmentation.

Ese experimento fue `MLP_3_dropout_batchnorm_weightdecay`, con accuracy media de 0.618.

### ¿Notaste cambios en la loss de entrenamiento al usar `BatchNorm`?

Sí. En general, BatchNorm tiende a estabilizar el entrenamiento y puede suavizar la evolución de las métricas. En los resultados agregados, se observó que el modelo con BatchNorm tuvo mejor validación que el modelo con Dropout solo.

---

# 8. Inicialización de parámetros

## Preguntas teóricas

### ¿Por qué es importante la inicialización de los pesos en una red neuronal?

Porque la inicialización determina el punto de partida de la optimización. Una mala inicialización puede producir gradientes muy pequeños, gradientes muy grandes, convergencia lenta o inestabilidad.

### ¿Qué podría ocurrir si todos los pesos se inicializan con el mismo valor?

Todas las neuronas de una misma capa aprenderían lo mismo, porque recibirían gradientes iguales. Esto rompe la capacidad del modelo de aprender representaciones diversas. Por eso se inicializan los pesos con valores aleatorios.

### ¿Cuál es la diferencia entre las inicializaciones de Xavier y He?

Xavier busca mantener estable la varianza de las activaciones entre capas y suele ser adecuada para activaciones simétricas como `tanh`.

He/Kaiming está pensada para redes con ReLU, teniendo en cuenta que ReLU anula aproximadamente la mitad de las activaciones negativas.

### ¿Por qué en una red con ReLU suele usarse la inicialización de He?

Porque ReLU deja pasar solo valores positivos y anula los negativos. He/Kaiming ajusta la varianza inicial de los pesos para compensar ese comportamiento y mantener una propagación más estable de las activaciones.

### ¿Qué capas de una red requieren inicialización explícita y cuáles no?

Principalmente las capas con parámetros entrenables, como `Linear` o `Conv2d`. Capas como `ReLU`, `Flatten` o `Dropout` no tienen pesos entrenables. BatchNorm sí tiene parámetros, pero PyTorch ya los inicializa de forma estándar.

---

## Actividades de modificación

### 1. Agregar inicialización manual

Se implementó inicialización manual para capas `Linear`, permitiendo elegir entre:

* `default`;
* `xavier`;
* `he`;
* `uniform`.

Además, los bias se inicializaron en cero.

### 2. Probar distintas estrategias de inicialización

Se probaron:

| Experimento                  | Inicialización | Accuracy media |
| ---------------------------- | -------------- | -------------: |
| MLP_5_xavier_initialization  | Xavier         |          0.570 |
| MLP_6_he_initialization      | He/Kaiming     |          0.555 |
| MLP_7_uniform_initialization | Uniforme       |          0.565 |

Ninguna inicialización manual superó a `MLP_3_dropout_batchnorm_weightdecay`.

### 3. Visualizar pesos en TensorBoard

Se agregaron histogramas de pesos en TensorBoard mediante `writer.add_histogram(...)`. Esto permite observar la distribución de pesos durante el entrenamiento.

---

## Preguntas prácticas

### ¿Qué diferencias notaste en la convergencia del modelo según la inicialización?

Las inicializaciones manuales no mejoraron el desempeño respecto de la mejor configuración con inicialización default. Xavier fue la mejor de las tres inicializaciones manuales probadas, pero quedó por debajo de `MLP_3`.

### ¿Alguna inicialización provocó inestabilidad, pérdida muy alta o NaNs?

No se observaron problemas severos como NaNs. Sin embargo, las inicializaciones manuales no mejoraron la validación media y produjeron resultados inferiores al mejor experimento.

### ¿Qué impacto tiene la inicialización sobre las métricas de validación?

Tuvo impacto medible, pero no positivo respecto de la mejor configuración. Las accuracies medias estuvieron aproximadamente entre 0.555 y 0.570, por debajo del mejor modelo MLP, que alcanzó 0.618.

### ¿Por qué `bias` se suele inicializar en cero?

Porque el sesgo no necesita romper simetría entre neuronas. La simetría se rompe con los pesos aleatorios. Inicializar bias en cero es simple, estable y usualmente suficiente.
