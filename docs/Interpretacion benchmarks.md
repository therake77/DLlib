# Interpretacion de graficas de benchmarks

Este documento interpreta las graficas generadas por:

```powershell
scripts\.venv\Scripts\python.exe scripts\analyze_benchmarks.py benchmarks\results.json --out-dir benchmarks\results\processed
```

Las graficas viven en:

```text
benchmarks/results/processed/figures/
```

La metrica principal de las graficas de speedup es:

```text
speedup_cpu_over_gpu = tiempo_CPU / tiempo_GPU
```

Por lo tanto:

- `speedup > 1`: GPU fue mas rapido que CPU.
- `speedup = 1`: CPU y GPU tardaron aproximadamente lo mismo.
- `speedup < 1`: CPU fue mas rapido que GPU.

Los valores interpretados aqui corresponden a `benchmarks/results/processed/speedups_cpu_gpu.csv` y `benchmarks/results/processed/transfer_bandwidth.csv`.

La CNN usada en los benchmarks de red completa no incluye BatchNorm. Se excluye temporalmente porque esa capa todavia esta bajo revision funcional en entrenamiento.

## `tensor_add_speedup.png`

![Speedup de tensor_add](../benchmarks/results/processed/figures/tensor_add_speedup.png)

Esta grafica compara una suma elemento a elemento en tensores de 4096, 65536 y 1048576 elementos.

Resultados observados:

```text
4096 elementos:    speedup 0.015
65536 elementos:   speedup 0.097
1048576 elementos: speedup 0.276
```

La CPU gana en todos los tamanos medidos. Es esperable para una operacion muy simple y con poca intensidad aritmetica. El costo fijo de lanzar un kernel CUDA y sincronizar domina frente a una sola suma por elemento.

Aunque el speedup mejora con el tamano, todavia no supera `1`. Para que este tipo de operacion favorezca a GPU, harian falta tensores mayores, fusion de kernels o menor frecuencia de sincronizacion.

## `matmul_speedup.png`

![Speedup de matmul](../benchmarks/results/processed/figures/matmul_speedup.png)

Esta grafica compara multiplicacion de matrices cuadradas:

```text
64x64:   speedup 0.448
128x128: speedup 1.744
256x256: speedup 5.104
```

La GPU pierde en `64x64`, pero gana desde `128x128` y mejora claramente en `256x256`. Esta es una de las graficas mas utiles para el paper porque muestra el punto donde el paralelismo de GPU empieza a compensar el overhead.

## `conv2d_forward_speedup.png`

![Speedup de conv2d_forward](../benchmarks/results/processed/figures/conv2d_forward_speedup.png)

Esta grafica compara el forward de una capa convolucional:

```text
batch=1,  C_in=1, C_out=8,  H=28: speedup 0.299
batch=8,  C_in=1, C_out=8,  H=28: speedup 0.841
batch=16, C_in=8, C_out=16, H=28: speedup 3.927
```

La CPU gana para el caso pequeno y queda casi empatada en batch 8. En el caso mas grande, la GPU gana con claridad.

Esto indica que la convolucion via `im2col + matmul` necesita suficiente trabajo acumulado para amortizar conversiones, lanzamientos de kernels y sincronizaciones. La grafica justifica variar batch size y dimensiones en lugar de reportar un unico caso.

## `fully_connected_forward_speedup.png`

![Speedup de fully_connected_forward](../benchmarks/results/processed/figures/fully_connected_forward_speedup.png)

Esta grafica compara el forward de capas densas:

```text
batch=1,  784->128:  speedup 0.087
batch=32, 784->128:  speedup 2.679
batch=64, 1024->256: speedup 8.116
```

Con batch 1, CPU es mucho mas rapida. Con batch 32 y batch 64, GPU gana con fuerza. Esto es coherente con la naturaleza de las capas densas: son multiplicaciones de matrices, y la GPU mejora cuando el problema tiene suficiente tamano.

## `cnn_forward_speedup.png`

![Speedup de cnn_forward](../benchmarks/results/processed/figures/cnn_forward_speedup.png)

Esta grafica mide el forward completo de la CNN tipo MNIST sin BatchNorm:

```text
batch=1:  speedup 1.196
batch=8:  speedup 2.326
batch=16: speedup 2.443
batch=32: speedup 2.457
```

La GPU gana en todos los batch sizes medidos. El beneficio aumenta al pasar de batch 1 a batch 8 y luego se estabiliza alrededor de `2.4x`.

Este resultado es mas representativo para inferencia que las capas aisladas, porque combina convolucion, activacion, pooling, flatten, capas densas y softmax.

## `cnn_training_step_same_batch_speedup.png`

![Speedup de cnn_training_step_same_batch](../benchmarks/results/processed/figures/cnn_training_step_same_batch_speedup.png)

Esta grafica mide una iteracion completa de entrenamiento con batch fijo:

```text
forward + backward + update_weights
```

Resultados observados:

```text
batch=1:  speedup 1.444
batch=8:  speedup 4.981
batch=16: speedup 6.318
batch=32: speedup 6.903
```

La GPU gana en todos los casos, especialmente desde batch 8. El beneficio crece hasta batch 32 y es mayor que en `cnn_forward` porque el paso de entrenamiento contiene mas trabajo computacional: backward, calculo de gradientes y actualizacion de pesos.

Este benchmark sigue la idea de `--same-batch`: el input y los labels se crean una sola vez y se reutilizan en cada iteracion. Por eso mide el costo computacional del paso de entrenamiento, no el costo de leer dataset ni avanzar por batches distintos.

## `tensor_transfer_bandwidth.png`

![Bandwidth de tensor_transfer](../benchmarks/results/processed/figures/tensor_transfer_bandwidth.png)

Esta grafica mide ancho de banda efectivo de transferencia entre CPU y GPU.

Resultados observados:

```text
CPU -> GPU, 4096 elementos:    0.156 GiB/s
CPU -> GPU, 65536 elementos:   2.406 GiB/s
CPU -> GPU, 1048576 elementos: 2.028 GiB/s

GPU -> CPU, 4096 elementos:    0.080 GiB/s
GPU -> CPU, 65536 elementos:   0.992 GiB/s
GPU -> CPU, 1048576 elementos: 2.826 GiB/s
```

Las transferencias pequenas tienen bajo bandwidth efectivo porque el overhead fijo pesa mucho. En tamanos mayores el throughput mejora. Esta grafica sirve para explicar por que conviene separar benchmarks compute-only de benchmarks end-to-end con transferencia.

## Lectura general

Los resultados muestran un patron consistente:

- GPU pierde o empata en problemas pequenos y muy simples.
- GPU mejora al aumentar batch size, canales, dimensiones o cantidad de operaciones.
- Las operaciones elementwise simples no justifican por si solas el uso de GPU en estos tamanos.
- Las operaciones tipo matriz, convolucion y entrenamiento completo son las que mas se benefician.
- Las transferencias CPU/GPU tienen overhead visible, especialmente en tamanos pequenos.
- El benchmark `same_batch` permite medir entrenamiento de forma controlada mientras se evita ruido de dataset.

## Precauciones

Estas graficas no deben interpretarse como rendimiento final absoluto del proyecto. Son una foto de una configuracion concreta:

- build `Release`;
- una maquina especifica;
- datos sinteticos para los benchmarks de red;
- red completa sin BatchNorm;
- medicion compute-only para operaciones y CNN;
- medicion separada para transferencias CPU/GPU;
- sincronizaciones internas despues de kernels CUDA;
- implementaciones actuales de CPU y GPU.

Para resultados de paper, conviene reportar tambien:

- numero de repeticiones;
- `--benchmark_min_time` usado;
- media o mediana usada;
- hardware CPU/GPU;
- compilador y version de CUDA;
- si el benchmark incluye o excluye transferencias CPU/GPU;
- batch sizes y formas exactas.
