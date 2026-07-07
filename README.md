# llcnn

`llcnn` es una biblioteca de redes neuronales convolucionales escrita desde cero en C++17 (CPU y CUDA opcional). No depende de ningún framework de machine learning: los tensores, las capas, los kernels matemáticos y el pipeline de datos estilo MNIST están implementados a mano.

Incluye:

- **Tensores** de rango fijo (hasta 4 dimensiones) con soporte de vistas no contiguas, broadcasting y strides.
- **Abstracción de dispositivo** (`CpuDevice` / `GpuDevice`), cada uno con su implementación de `Math` (`CpuMath` / `GpuMath`), de modo que las capas se escriben contra la interfaz y no contra CPU o CUDA directamente.
- **Capas**: `Conv2D`, `FullyConnected`, `BatchNorm`, `ActivationLayer`, `MaxPooling` / `AvgPooling`, `Flatten` y `OutputLayer` (softmax + entropía cruzada).
- Convolución vía im2col, pooling, batchnorm y backprop implementados manualmente.
- Un **pipeline de datos** simple (`Dataset` / `DataLoader` / `FlatDataset`) para cargar imágenes y generar batches.

El soporte CUDA se detecta automáticamente en CMake: si hay un compilador CUDA disponible se compilan las rutas de GPU; si no, simplemente se excluyen del build.

## Compilación

Configurado con CMake (generador MSVC / Visual Studio 17 2022; build en modo Debug por defecto).

```
cmake -S . -B build
cmake --build build --config Debug
```

## Ejecutar `project/app.cpp`

`app.cpp` es el ejemplo de referencia: arma una CNN, carga MNIST, entrena y luego hace inferencia sobre algunas imágenes. Se compila en el ejecutable `main`.

1. Prepara los datos de MNIST en formato binario plano. `app.cpp` espera encontrarlos en `datasets/mnist/flat/`:

   ```
   datasets/mnist/flat/train_images_float32.bin
   datasets/mnist/flat/train_labels_uint8.bin
   ```

   Puedes generarlos a partir de los archivos IDX de MNIST con el script incluido:

   ```
   python scripts/export_mnist_flat.py
   ```

2. Compila el ejecutable:

   ```
   cmake --build build --config Debug --target main
   ```

3. Ejecútalo desde la raíz del proyecto (las rutas del dataset son relativas a este directorio):

   ```
   ./build/Debug/main.exe
   ```

### Opciones de línea de comandos

```
main [--cpu|--gpu] [--epochs N] [--batches N] [--batch-size N] [--lr LR] [--same-batch] [--reset-each-epoch]
```

| Opción               | Descripción                                             | Por defecto |
| -------------------- | ------------------------------------------------------- | ----------- |
| `--cpu` / `--gpu`    | Elige el dispositivo de ejecución.                      | `--cpu`     |
| `--epochs N`         | Número de épocas de entrenamiento.                      | `10`        |
| `--batches N`        | Batches por época.                                      | `937`       |
| `--batch-size N`     | Tamaño de cada batch.                                   | `64`        |
| `--lr LR`            | Tasa de aprendizaje.                                    | `-0.01`     |
| `--same-batch`       | Reutiliza siempre el primer batch (útil para depurar).  | off         |
| `--reset-each-epoch` | Reinicia el índice del dataset al inicio de cada época. | off         |

### Parámetros recomendados

La configuración probada que da buenos resultados es:

```
./build/Debug/main.exe --batches 100 --batch-size 64 --lr -0.02
```

Ejemplo (entrenar en GPU durante 5 épocas):

```
./build/Debug/main.exe --gpu --epochs 5
```

Al terminar el entrenamiento, el programa realiza inferencia sobre unas imágenes y muestra la tasa de aciertos (*hit rate*).