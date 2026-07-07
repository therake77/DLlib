# Nsight Systems y Nsight Compute

Este documento describe como usar Nsight Systems y Nsight Compute sobre `cnn_benchmark`.

La idea es usar estas herramientas como complemento de Google Benchmark:

- Google Benchmark produce los tiempos para graficas de speedup, eficiencia y tiempo promedio.
- Nsight Systems muestra la linea de tiempo CPU/GPU, transferencias, kernels y sincronizaciones.
- Nsight Compute muestra metricas internas de kernels CUDA concretos.

## Estado local de herramientas

En esta maquina `ncu` ya esta disponible desde PowerShell:

```powershell
ncu --version
```

Nsight Systems esta instalado, pero `nsys` no esta en el `PATH`. Se puede ejecutar usando la ruta completa:

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe" --version
```

Si se quiere usar solo `nsys`, agregar esta carpeta al `PATH`:

```text
C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64
```

## Preparacion

Compilar en Release:

```powershell
cmake -S . -B build
cmake --build build --target cnn_benchmark --config Release
```

Crear carpetas para reportes:

```powershell
New-Item -ItemType Directory -Force benchmarks\results\nsys
New-Item -ItemType Directory -Force benchmarks\results\ncu
```

Para profiling conviene medir un benchmark a la vez. Perfilar toda la suite completa produce reportes grandes y dificiles de interpretar.

## Nsight Systems

Usar Nsight Systems primero para responder preguntas de alto nivel:

- cuanto tiempo pasa la aplicacion en CPU;
- cuando se lanzan kernels CUDA;
- si hay huecos entre kernels;
- si hay sincronizaciones costosas;
- si las copias CPU/GPU dominan el tiempo;
- si el benchmark realmente esta ejecutando trabajo en GPU.

Ejemplo para el entrenamiento GPU con batch 64:

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe" profile `
  --trace=cuda,nvtx `
  --sample=none `
  --cpuctxsw=none `
  --force-overwrite=true `
  -o benchmarks\results\nsys\cnn_training_step_gpu_b64 `
  .\build\benchmarks\Release\cnn_benchmark.exe `
  --benchmark_filter="^cnn_training_step/GPU/64$" `
  --benchmark_repetitions=1 `
  --benchmark_min_time=0.1s
```

Ejemplo para un loop de 10 pasos de entrenamiento:

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe" profile `
  --trace=cuda,nvtx `
  --sample=none `
  --cpuctxsw=none `
  --force-overwrite=true `
  -o benchmarks\results\nsys\cnn_training_loop_gpu_b64_s10 `
  .\build\benchmarks\Release\cnn_benchmark.exe `
  --benchmark_filter="^cnn_training_loop/GPU/64/10$" `
  --benchmark_repetitions=1 `
  --benchmark_min_time=0.1s
```

Ejemplo para `cnn_forward` GPU con batch 64:

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe" profile `
  --trace=cuda,nvtx `
  --sample=none `
  --cpuctxsw=none `
  --force-overwrite=true `
  -o benchmarks\results\nsys\cnn_forward_gpu_b64 `
  .\build\benchmarks\Release\cnn_benchmark.exe `
  --benchmark_filter="^cnn_forward/GPU/64$" `
  --benchmark_repetitions=1 `
  --benchmark_min_time=0.1s
```

Abrir el reporte generado con la interfaz grafica de Nsight Systems:

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\host-windows-x64\nsys-ui.exe" benchmarks\results\nsys\cnn_training_step_gpu_b64.nsys-rep
```

## Nsight Compute

Usar Nsight Compute despues de Nsight Systems, cuando ya se sabe que kernel o benchmark vale la pena investigar.

Preguntas que responde:

- ocupacion teorica y alcanzada;
- uso de memoria global;
- throughput de memoria;
- throughput de SM;
- divergencia o baja utilizacion;
- kernels limitados por memoria o por computo.

Ejemplo para convolucion GPU:

```powershell
ncu `
  --target-processes all `
  --set full `
  --kernel-name-base demangled `
  --launch-skip 5 `
  --launch-count 10 `
  --force-overwrite `
  --export benchmarks\results\ncu\conv2d_forward_gpu_large `
  .\build\benchmarks\Release\cnn_benchmark.exe `
  --benchmark_filter="^conv2d_forward/GPU/16/8/16/28$" `
  --benchmark_repetitions=1 `
  --benchmark_min_time=0.05s
```

Ejemplo para multiplicacion de matrices GPU:

```powershell
ncu `
  --target-processes all `
  --set full `
  --kernel-name-base demangled `
  --launch-skip 5 `
  --launch-count 10 `
  --force-overwrite `
  --export benchmarks\results\ncu\matmul_gpu_256 `
  .\build\benchmarks\Release\cnn_benchmark.exe `
  --benchmark_filter="^matmul/GPU/256/256/256$" `
  --benchmark_repetitions=1 `
  --benchmark_min_time=0.05s
```

Abrir un reporte de Nsight Compute:

```powershell
ncu-ui benchmarks\results\ncu\conv2d_forward_gpu_large.ncu-rep
```

## Que guardar para el paper

Para el informe, guardar:

- tablas y graficas desde `benchmarks/results.json` y `scripts/analyze_benchmarks.py`;
- capturas de Nsight Systems mostrando kernels, copias y sincronizaciones;
- una tabla corta de Nsight Compute para 1 o 2 kernels importantes.

No es necesario incluir todos los reportes de Nsight en el paper. Lo ideal es usar Nsight para justificar por que GPU acelera mejor en batches grandes y por que en problemas pequenos el overhead domina.

## Siguiente mejora recomendada

Agregar rangos NVTX al codigo permitiria que Nsight Systems muestre nombres como:

```text
Conv2D forward
Pooling forward
FullyConnected forward
CNN training step
```

Sin NVTX, Nsight Systems igual funciona, pero la interpretacion queda mas dependiente de nombres de kernels y llamadas CUDA.
