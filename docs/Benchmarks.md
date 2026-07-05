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

## Criterio para benchmarks end-to-end

Un benchmark end-to-end de CNN CPU vs GPU debe agregarse solo cuando:

- las operaciones usadas por la red tienen implementacion CPU y GPU;
- existen tests que comparan resultados CPU vs GPU;
- la ruta GPU no usa funciones vacias o no implementadas;
- se decide explicitamente si el tiempo incluye o excluye transferencias CPU/GPU;
- el benchmark no depende de impresion por consola ni lectura de archivos dentro del bucle medido.

Hasta entonces, es preferible medir operaciones y capas por separado.
