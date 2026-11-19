# DLlib

`DLLib` es una biblioteca de redes neuronales convolucionales escrita desde cero en C++17 (CPU y CUDA opcional). No depende de ningún framework de machine learning: los tensores, las capas, los kernels matemáticos y el pipeline de datos estilo MNIST están implementados a mano.

`DLLib` is a library written from scratch in C++17 focused on implementing basic deep learning primitives, such as Dense Layers and Convolution Layers, etc. It also provide ways to work with multidimensional arrays (tensors), including broadcasting, and support routines written in CPU and GPU, using a device abstraction layer.


## Compilation

You need to use `cmake`
```
cmake -S . -B build
cmake --build build --config Debug
```

## App example

There is an example of a CNN trained over MNIST on `project/app.cpp`. You can execute it following the next steps

1. You need the mnist files preprocessed as `.bin` files.

   ```
   datasets/mnist/flat/train_images_float32.bin
   datasets/mnist/flat/train_labels_uint8.bin
   ```

   Files can be generated using the next python script:

   ```
   python scripts/export_mnist_flat.py
   ```

2. Compile the executable:

   ```
   cmake --build build --config Debug --target main
   ```

3. Execute it from the root directory:

   ```
   ./build/Debug/main.exe
   ```

### Command line arguments

```
main [--cpu|--gpu] [--epochs N] [--batches N] [--batch-size N] [--lr LR] [--same-batch] [--reset-each-epoch]
```

| Option               | Description                                             | Default     |
| -------------------- | ------------------------------------------------------- | ----------- |
| `--cpu` / `--gpu`    | Switch between CPU and GPU                              | `--cpu`     |
| `--epochs N`         | Number of epochs                                        | `10`        |
| `--batches N`        | Batches per epoch                                       | `937`       |
| `--batch-size N`     | Batch size                                              | `64`        |
| `--lr LR`            | Learning rate.                                          | `-0.01`     |


### Recommended hyperparameters

I tested this hyperparameter configuration:

```
./build/Debug/main.exe --batches 100 --batch-size 64 --lr -0.02
```

And this execution

```
./build/Debug/main.exe --gpu --epochs 5
```

Resulting in a hit rate of 99%.