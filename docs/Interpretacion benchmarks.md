# Interpretacion de graficas de benchmarks

Este documento interpreta las graficas generadas por:

```powershell
scripts\.venv\Scripts\python.exe scripts\analyze_benchmarks.py benchmarks\results.json --out-dir benchmarks\results\processed
```

Las graficas viven en:

```text
benchmarks/results/processed/figures/
```

La metrica principal es:

```text
speedup_cpu_over_gpu = tiempo_CPU / tiempo_GPU
```

Por lo tanto:

- `speedup > 1`: GPU fue mas rapido que CPU.
- `speedup = 1`: CPU y GPU tardaron aproximadamente lo mismo.
- `speedup < 1`: CPU fue mas rapido que GPU.

Los valores interpretados aqui corresponden al archivo `benchmarks/results/processed/speedups_cpu_gpu.csv` generado a partir de `benchmarks/results.json`.

## `tensor_add_speedup.png`

![Speedup de tensor_add](../benchmarks/results/processed/figures/tensor_add_speedup.png)

Esta grafica compara una suma elemento a elemento en tensores de 4096, 65536 y 1048576 elementos.

Resultados observados:

```text
4096 elementos:    speedup 0.016
65536 elementos:   speedup 0.097
1048576 elementos: speedup 0.271
```

Interpretacion:

La CPU gana en todos los tamanos medidos. Esto es esperable para una operacion muy simple y con poca intensidad aritmetica: cada elemento hace una suma y poco mas. El costo fijo de lanzar un kernel CUDA y sincronizar puede dominar el tiempo total, especialmente porque las funciones GPU actuales sincronizan despues de cada kernel.

Aunque el speedup mejora al aumentar el tamano, todavia no supera `1`. Para este tipo de operacion, la GPU necesita tamanos mucho mayores, fusion de kernels o evitar sincronizaciones frecuentes para compensar el overhead.

## `matmul_speedup.png`

![Speedup de matmul](../benchmarks/results/processed/figures/matmul_speedup.png)

Esta grafica compara multiplicacion de matrices cuadradas:

```text
64x64:   speedup 0.502
128x128: speedup 1.738
256x256: speedup 4.686
```

Interpretacion:

La GPU pierde en `64x64`, pero empieza a ganar desde `128x128` y escala mejor en `256x256`. Este patron es el esperado para operaciones con mayor intensidad aritmetica: a medida que crece el trabajo por lanzamiento de kernel, el overhead fijo pesa menos y el paralelismo de la GPU empieza a dominar.

Esta grafica es una buena candidata para el informe porque muestra claramente el punto de cruce donde usar GPU empieza a valer la pena.

## `conv2d_forward_speedup.png`

![Speedup de conv2d_forward](../benchmarks/results/processed/figures/conv2d_forward_speedup.png)

Esta grafica compara el forward de una capa convolucional:

```text
batch=1, C_in=1, C_out=8,  H=28: speedup 0.320
batch=8, C_in=1, C_out=8,  H=28: speedup 0.920
batch=16,C_in=8, C_out=16, H=28: speedup 3.679
```

Interpretacion:

La CPU gana para el caso pequeno y la GPU queda cerca de empatar para batch 8. En el caso mas grande, la GPU gana con claridad.

Esto refleja que la convolucion implementada via `im2col + matmul` necesita suficiente trabajo acumulado para amortizar conversiones, llamadas a kernels y sincronizaciones. Con pocos canales y batch pequeno, la CPU optimizada puede ser mas eficiente. Con mas batch, canales y filtros, la GPU aprovecha mejor el paralelismo.

Esta grafica ayuda a justificar por que las mediciones deben variar batch size y dimensiones, no solo reportar un unico caso.

## `fully_connected_forward_speedup.png`

![Speedup de fully_connected_forward](../benchmarks/results/processed/figures/fully_connected_forward_speedup.png)

Esta grafica compara el forward de capas densas:

```text
batch=1,  784->128:  speedup 0.100
batch=32, 784->128:  speedup 2.895
batch=64, 1024->256: speedup 8.617
```

Interpretacion:

Con batch 1, la CPU es claramente mas rapida. El trabajo es demasiado pequeno para justificar el overhead de la GPU.

Al subir a batch 32 y luego a una capa mas grande con batch 64, la GPU pasa a ser mucho mas rapida. Esto es coherente con la naturaleza de las capas densas: internamente son multiplicaciones de matrices, y la GPU mejora cuando el problema tiene suficiente tamano.

Esta grafica es una de las mas claras para explicar speedup por aumento de granularidad.

## `cnn_forward_speedup.png`

![Speedup de cnn_forward](../benchmarks/results/processed/figures/cnn_forward_speedup.png)

Esta grafica mide el forward completo de la CNN tipo MNIST:

```text
batch=1: speedup 1.045
batch=8: speedup 2.043
```

Interpretacion:

Con batch 1, CPU y GPU estan casi empatadas. La GPU apenas gana, pero el margen es pequeno. Con batch 8, la GPU alcanza aproximadamente `2x`.

Esto confirma que, al medir una red completa, el beneficio GPU aparece con batches mas grandes. Tambien muestra que el speedup end-to-end es menor que el de algunas capas individuales grandes, porque la red incluye operaciones con menor intensidad aritmetica, buffers temporales, reshapes, `im2col`, pooling, BatchNorm y sincronizaciones.

Esta grafica es util para presentar el rendimiento global de inferencia.

## `cnn_training_step_same_batch_speedup.png`

![Speedup de cnn_training_step_same_batch](../benchmarks/results/processed/figures/cnn_training_step_same_batch_speedup.png)

Esta grafica mide una iteracion completa de entrenamiento con batch fijo:

```text
forward + backward + update_weights
```

Resultados observados:

```text
batch=1: speedup 1.127
batch=8: speedup 3.441
```

Interpretacion:

La GPU gana levemente con batch 1 y gana claramente con batch 8. El beneficio es mayor que en `cnn_forward`, porque el paso de entrenamiento incluye mas trabajo computacional: backward, gradientes y actualizacion de pesos.

Este benchmark sigue la idea de `--same-batch`: se reutiliza el mismo input y los mismos labels en cada iteracion. Eso es importante porque aisla el costo de computo de la red sin mezclar lectura de dataset, cambios de indice ni ruido por impresion en consola.

Para el informe, esta grafica puede presentarse como medicion de entrenamiento controlado, no como medicion completa de entrenamiento sobre todo MNIST.

## Lectura general

Los resultados muestran un patron consistente:

- GPU pierde o empata en problemas pequenos.
- GPU mejora al aumentar batch size, canales, dimensiones o cantidad de operaciones.
- Las operaciones elementwise simples no justifican por si solas el uso de GPU en estos tamanos.
- Las operaciones tipo matriz/convolucion son las que mas se benefician.
- El entrenamiento con `same_batch` permite medir el costo computacional del paso de entrenamiento sin ruido de dataset.

## Precauciones

Estas graficas no deben interpretarse como rendimiento final absoluto del proyecto. Son una foto de una configuracion concreta:

- build `Release`;
- una maquina especifica;
- datos sinteticos para los benchmarks de red;
- medicion compute-only, sin lectura de archivos dentro del loop;
- sincronizaciones internas despues de kernels CUDA;
- implementaciones actuales de CPU y GPU.

Para resultados de paper, conviene reportar tambien:

- numero de repeticiones;
- media o mediana usada;
- hardware CPU/GPU;
- compilador y version de CUDA;
- si el benchmark incluye o excluye transferencias CPU/GPU;
- batch sizes y formas exactas.
