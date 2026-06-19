# Resumen final del trabajo

## Objetivo

El objetivo del trabajo fue resolver un problema de clasificación multiclase de lesiones dermatológicas. Primero se trabajó con una red MLP como modelo base y luego se resolvió nuevamente el problema utilizando redes convolucionales. En la Parte 2 se buscó maximizar el accuracy mediante variaciones de data augmentation, técnicas de regularización, búsqueda de hiperparámetros y transfer learning.

## Preparación del dataset

Antes de entrenar los modelos se revisó la existencia de imágenes duplicadas. Se generó un split limpio y estratificado, dejando un conjunto de test fijo separado desde el comienzo. El archivo utilizado para todos los experimentos fue:

```text
data/splits/final_split_5fold.csv
```

La partición final quedó organizada de la siguiente manera:

| Subconjunto      | Cantidad de imágenes |
| ---------------- | -------------------: |
| Train/validation |                  717 |
| Test             |                  127 |
| Total limpio     |                  844 |

La selección de hiperparámetros se realizó usando validación cruzada de 5 folds sobre `trainval`. El conjunto de test no se utilizó para elegir modelos ni hiperparámetros.

## Parte 1: MLP

Se entrenó una MLP como modelo base. Se probaron variantes con dropout, batch normalization, weight decay, data augmentation, distintos inicializadores y learning rates.

El mejor modelo MLP obtenido por validación cruzada fue:

```text
MLP_10_lower_lr_dropout03_bn_wd
```

Configuración principal:

| Hiperparámetro      | Valor      |
| ------------------- | ---------- |
| Arquitectura        | MLP        |
| Hidden dims         | [512, 128] |
| Dropout             | 0.3        |
| Batch normalization | Sí         |
| Weight decay        | 1e-4       |
| Learning rate       | 3e-4       |
| Augmentation        | minimal    |
| Inicialización      | He         |

Resultado en validación cruzada:

| Modelo                          | Val accuracy mean |
| ------------------------------- | ----------------: |
| MLP_10_lower_lr_dropout03_bn_wd |            0.6136 |

Resultado final en test:

| Modelo    | Test accuracy | Test macro F1 | Test balanced accuracy |
| --------- | ------------: | ------------: | ---------------------: |
| MLP final |        0.5984 |        0.5789 |                 0.6000 |

## Parte 2: CNN entrenada desde cero

Luego se reemplazó la MLP por una CNN inspirada en AlexNet. La red utiliza bloques convolucionales, ReLU, max pooling, batch normalization y dropout en el clasificador.

Primero se entrenó una CNN baseline:

```text
CNN_0_alexnetsmall_bn_dropout_wd_lightaug
```

Luego se probaron variantes modificando:

* Batch normalization;
* Dropout;
* Weight decay;
* Learning rate;
* Data augmentation;
* Capacidad del modelo.

El mejor modelo CNN desde cero fue:

```text
CNN_FULL_4_aug_minimal
```

Configuración principal:

| Hiperparámetro      | Valor        |
| ------------------- | ------------ |
| Arquitectura        | AlexNetSmall |
| Image size          | 128          |
| Batch size          | 32           |
| Dropout             | 0.3          |
| Batch normalization | Sí           |
| Weight decay        | 1e-4         |
| Learning rate       | 3e-4         |
| Augmentation        | minimal      |

Resultado en validación cruzada:

| Modelo                 | Val accuracy mean | Val macro F1 mean | Val balanced accuracy mean |
| ---------------------- | ----------------: | ----------------: | -------------------------: |
| CNN_FULL_4_aug_minimal |            0.6792 |            0.6739 |                     0.6732 |

Resultado final en test:

| Modelo    | Test accuracy | Test macro F1 | Test balanced accuracy |
| --------- | ------------: | ------------: | ---------------------: |
| CNN final |        0.6929 |        0.6875 |                 0.6870 |

La CNN entrenada desde cero superó claramente a la MLP, lo cual es esperable porque las capas convolucionales aprovechan la estructura espacial de las imágenes.

## Bonus: transfer learning

Como bonus se probó transfer learning usando ResNet18 preentrenada en ImageNet. Se compararon estrategias con la red congelada y con fine tuning parcial.

La mejor estrategia fue:

```text
TL_FULL_3_resnet18_layer4_minimal
```

Configuración principal:

| Hiperparámetro | Valor                                      |
| -------------- | ------------------------------------------ |
| Modelo         | ResNet18 pretrained                        |
| Estrategia     | Fine tuning de layer4 + clasificador final |
| Image size     | 224                                        |
| Batch size     | 16                                         |
| Dropout        | 0.3                                        |
| Weight decay   | 1e-4                                       |
| Learning rate  | 1e-4                                       |
| Augmentation   | minimal                                    |

Resultado en validación cruzada:

| Modelo                            | Val accuracy mean | Val macro F1 mean | Val balanced accuracy mean |
| --------------------------------- | ----------------: | ----------------: | -------------------------: |
| TL_FULL_3_resnet18_layer4_minimal |            0.8298 |            0.8371 |                     0.8375 |

Resultado final en test:

| Modelo                  | Test accuracy | Test macro F1 | Test balanced accuracy |
| ----------------------- | ------------: | ------------: | ---------------------: |
| Transfer learning final |        0.8583 |        0.8635 |                 0.8667 |

## Comparación final en test

Para la evaluación final, cada modelo elegido fue reentrenado usando todo `trainval` y luego evaluado una única vez sobre `test`.

| Modelo            | Test accuracy | Test macro F1 | Test balanced accuracy |
| ----------------- | ------------: | ------------: | ---------------------: |
| MLP               |        0.5984 |        0.5789 |                 0.6000 |
| CNN desde cero    |        0.6929 |        0.6875 |                 0.6870 |
| Transfer learning |        0.8583 |        0.8635 |                 0.8667 |

El mejor modelo final fue ResNet18 con fine tuning del último bloque convolucional. Este modelo superó ampliamente tanto a la MLP como a la CNN entrenada desde cero.

## Registro de experimentos

Se dejaron registros de los experimentos en:

```text
runs/
mlruns/
experiments/
results/
```

En particular:

* `runs/` contiene los logs de TensorBoard;
* `mlruns/` contiene los experimentos de MLflow;
* `experiments/` contiene métricas, summaries y predicciones;
* `results/` contiene curvas de entrenamiento, matrices de confusión y classification reports.

Los directorios `runs/` y `mlruns/` no se suben a GitHub por tamaño. Se comprimen y se entregan por Drive.

## Conclusión

El uso de CNNs mejoró el desempeño respecto de la MLP, confirmando la ventaja de explotar la estructura espacial de las imágenes. La mejor CNN desde cero alcanzó un accuracy de test de 0.6929, mientras que la MLP alcanzó 0.5984. El mejor resultado global se obtuvo con transfer learning usando ResNet18 preentrenada y fine tuning del último bloque, alcanzando un accuracy de test de 0.8583.
