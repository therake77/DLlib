# Benchmarks

Este documento describe la estructura recomendada para crear benchmarks dentro del proyecto. La idea principal es medir piezas concretas de la CNN de forma reproducible, sin mezclar pruebas funcionales, impresion por consola, carga de datos o depuracion con el tiempo de computo que se quiere reportar.

## Estructura

Los benchmarks viven en el directorio:

```text
benchmarks/
```

La estructura minima es:

```text
benchmarks/
  CMakeLists.txt
  cnn_benchmark.cpp
```

El `CMakeLists.txt` raiz debe incluir:

```cmake
add_subdirectory(benchmarks)
```

El archivo `benchmarks/CMakeLists.txt` se encarga de:

- descargar o encontrar Google Benchmark;
- crear el ejecutable de benchmarks;
- enlazar `cnn_lib`;
- enlazar `benchmark::benchmark` y `benchmark::benchmark_main`;
- enlazar CUDA cuando el compilador CUDA este disponible.

## Tipos de benchmark

Conviene separar los benchmarks por nivel:

- operaciones base: `add`, `multiply`, `divide`, `matmul`, `im2col`, pooling;
- capas: `Conv2D::forward`, `FullyConnected::forward`, `BatchNorm::forward`;
- pipeline parcial: `Conv2D -> ReLU -> Pooling -> Flatten -> FullyConnected`;
- pipeline completo: CNN forward o entrenamiento, solo cuando todas las operaciones necesarias esten validadas en CPU y GPU;
- dataset: carga de batches, separado del computo de la red.

No es recomendable medir todo desde el inicio. Primero se miden operaciones pequenas y luego se sube de nivel.

## Patron recomendado

Cada benchmark debe seguir esta forma:

```cpp
void operation_name(benchmark::State& state, BenchDevice bench_device) {
    auto device = make_device(bench_device);

    // 1. Leer parametros desde state.range(...)
    // 2. Crear tensores, capas y buffers fuera del bucle
    // 3. Ejecutar un warmup fuera del bucle
    // 4. Medir solo la operacion dentro del bucle

    for(auto _ : state) {
        // operacion medida
        benchmark::DoNotOptimize(/* resultado */);
    }

    // 5. Registrar contadores
}
```

Los tensores, pesos, capas y buffers deben crearse fuera del bucle `for(auto _ : state)`. Si se crean dentro, el benchmark mide asignacion de memoria ademas del calculo.

## CPU y GPU

El proyecto usa una abstraccion de dispositivos. Por eso los benchmarks pueden compartir casi todo el codigo:

```cpp
enum class BenchDevice {
    CPU,
    GPU
};
```

La funcion auxiliar debe crear el dispositivo correcto:

```cpp
std::shared_ptr<Device> make_device(BenchDevice device);
```

Los benchmarks GPU deben compilarse solo cuando CUDA este disponible:

```cpp
#ifdef __USE_CUDA__
BENCHMARK(operation_gpu)->Name("operation/GPU");
#endif
```

Esto permite que el proyecto siga compilando en maquinas sin CUDA.

## Registro con Google Benchmark

Un benchmark se registra al final del archivo:

```cpp
BENCHMARK(matmul_cpu)
    ->Name("matmul/CPU")
    ->Args({64, 64, 64})
    ->Args({128, 128, 128});
```

Para GPU:

```cpp
#ifdef __USE_CUDA__
BENCHMARK(matmul_gpu)
    ->Name("matmul/GPU")
    ->Args({64, 64, 64})
    ->Args({128, 128, 128});
#endif
```

Mantener el mismo nombre base y los mismos argumentos para CPU y GPU facilita calcular speedup:

```text
matmul/CPU/128/128/128
matmul/GPU/128/128/128
```

## Contadores

Los contadores ayudan a producir tablas y graficas:

```cpp
state.counters["elements"] = static_cast<double>(elements);
state.counters["items_per_second"] = benchmark::Counter(
    static_cast<double>(elements),
    benchmark::Counter::kIsIterationInvariantRate
);
```

Para operaciones con conteo aproximado de FLOPs:

```cpp
state.counters["flop_per_second"] = benchmark::Counter(
    static_cast<double>(2 * m * n * k),
    benchmark::Counter::kIsIterationInvariantRate
);
```

Usar `kIsIterationInvariantRate` cuando el valor representa trabajo hecho por una iteracion del benchmark.

## Que medir

Para graficas de speedup y eficiencia, medir tiempos comparables:

- CPU secuencial vs CPU paralela;
- CPU vs GPU sin transferencias, cuando los datos ya estan en el dispositivo;
- CPU vs GPU con transferencias, si se quiere medir el costo end-to-end real;
- tiempo por capa;
- tiempo por tamano de entrada;
- tiempo por batch size.

La formula basica de speedup es:

```text
speedup = tiempo_base / tiempo_optimizado
```

Para eficiencia con CPU multihilo:

```text
eficiencia = speedup / numero_de_hilos
```

## Que no mezclar

Evitar mezclar en el mismo benchmark:

- carga de archivos;
- impresion por consola;
- inicializacion aleatoria;
- asignaciones repetidas de memoria;
- copias CPU/GPU, salvo que el benchmark tenga como objetivo medir transferencias;
- validacion exhaustiva de resultados.

La validacion debe vivir en tests. El benchmark debe asumir que la operacion ya fue validada.

## Ejecucion

Compilar:

```powershell
cmake -S . -B build
cmake --build build --target cnn_benchmark --config Release
```

Ejecutar todo:

```powershell
.\build\benchmarks\Release\cnn_benchmark.exe
```

Filtrar un grupo:

```powershell
.\build\benchmarks\Release\cnn_benchmark.exe --benchmark_filter=matmul
```

Usar repeticiones:

```powershell
.\build\benchmarks\Release\cnn_benchmark.exe --benchmark_repetitions=10 --benchmark_report_aggregates_only=true
```

Generar salida JSON:

```powershell
.\build\benchmarks\Release\cnn_benchmark.exe --benchmark_format=json --benchmark_out=benchmarks/results.json --benchmark_out_format=json
```

Flujo recomendado para generar datos reproducibles:

```powershell
cmake -S . -B build
cmake --build build --target cnn_benchmark --config Release
.\build\benchmarks\Release\cnn_benchmark.exe `
  --benchmark_min_time=0.02s `
  --benchmark_repetitions=10 `
  --benchmark_report_aggregates_only=false `
  --benchmark_format=json `
  --benchmark_out=benchmarks/results.json `
  --benchmark_out_format=json
```

## Analisis y graficas

El script de analisis vive en:

```text
scripts/analyze_benchmarks.py
```

Crear el entorno virtual con Python 3.13:

```powershell
python -m venv scripts\.venv
```

Instalar dependencias:

```powershell
scripts\.venv\Scripts\python.exe -m pip install -r scripts\requirements.txt
```

Analizar el JSON generado por Google Benchmark:

```powershell
scripts\.venv\Scripts\python.exe scripts\analyze_benchmarks.py benchmarks\results.json --out-dir benchmarks\results\processed
```

Salidas generadas:

```text
benchmarks/results/processed/benchmark_rows.csv
benchmarks/results/processed/speedups_cpu_gpu.csv
benchmarks/results/processed/transfer_bandwidth.csv
benchmarks/results/processed/figures/
```

`benchmark_rows.csv` contiene las filas limpias de Google Benchmark. `speedups_cpu_gpu.csv` contiene los pares CPU/GPU y la columna:

```text
speedup_cpu_over_gpu = cpu_real_time_s / gpu_real_time_s
```

Las figuras PNG muestran el speedup CPU/GPU por benchmark. Una barra mayor que `1` significa que GPU fue mas rapido que CPU para ese caso. Una barra menor que `1` significa que CPU fue mas rapido que GPU.

`transfer_bandwidth.csv` contiene las mediciones de transferencia `CPU_to_GPU` y `GPU_to_CPU`, reportadas como bytes por segundo y GiB/s. La figura `tensor_transfer_bandwidth.png` resume esos resultados.

Si solo se quieren generar CSV sin figuras:

```powershell
scripts\.venv\Scripts\python.exe scripts\analyze_benchmarks.py benchmarks\results.json --no-plots
```

Si se quiere usar la mediana en vez de la media:

```powershell
scripts\.venv\Scripts\python.exe scripts\analyze_benchmarks.py benchmarks\results.json --aggregate median
```

## Profiling con Nsight

Despues de generar los tiempos base con Google Benchmark, usar Nsight Systems y Nsight Compute para explicar el comportamiento observado:

- Nsight Systems: linea de tiempo CPU/GPU, kernels, transferencias y sincronizaciones.
- Nsight Compute: metricas internas de kernels CUDA concretos.

Los comandos recomendados estan documentados en:

```text
docs/Nsight profiling.md
```

## Criterio para benchmarks end-to-end

Un benchmark end-to-end de CNN CPU vs GPU debe agregarse solo cuando:

- las operaciones usadas por la red tienen implementacion CPU y GPU;
- existen tests que comparan resultados CPU vs GPU;
- la ruta GPU no usa funciones vacias o no implementadas;
- se decide explicitamente si el tiempo incluye o excluye transferencias CPU/GPU;
- el benchmark no depende de impresion por consola ni lectura de archivos dentro del bucle medido.

Hasta entonces, es preferible medir operaciones y capas por separado.

## Benchmarks de red completa

El ejecutable `cnn_benchmark` incluye benchmarks de una CNN tipo MNIST sin BatchNorm:

```text
Conv2D -> LeakyReLU -> MaxPooling
Conv2D -> LeakyReLU -> MaxPooling
Flatten -> FullyConnected -> LeakyReLU -> FullyConnected -> OutputLayer
```

BatchNorm se excluye de estos benchmarks porque la capa todavia esta bajo revision funcional en el entrenamiento. Hasta que esa ruta este estable, las graficas principales de red completa deben medir la arquitectura que se espera que aprenda correctamente.

Los benchmarks disponibles son:

```text
cnn_forward/CPU
cnn_forward/GPU
cnn_training_step_same_batch/CPU
cnn_training_step_same_batch/GPU
tensor_transfer/CPU_to_GPU
tensor_transfer/GPU_to_CPU
```

`cnn_training_step_same_batch` mide una iteracion completa:

```text
forward + backward + update_weights
```

Este benchmark sigue la misma idea del flag `--same-batch` de `project/app.cpp`: el input y los labels se crean una sola vez fuera del bucle de medicion y luego se reutiliza exactamente el mismo batch en cada iteracion. Esto evita mezclar el costo del entrenamiento con variacion de dataset, carga de archivos o avance del indice del batch.

Ejecutar solo estos benchmarks:

```powershell
.\build\benchmarks\Release\cnn_benchmark.exe --benchmark_filter=cnn_
```

Ejecutar solo entrenamiento con batch fijo:

```powershell
.\build\benchmarks\Release\cnn_benchmark.exe --benchmark_filter=cnn_training_step_same_batch
```
