# Paquetes de mejoras

Este documento resume los cambios aplicados durante la revision de rendimiento y estabilidad del entrenamiento. La intencion es dejar trazabilidad: que problema se observo, que se cambio, que archivos se tocaron y como se valido.

## Primer paquete de mejoras

Objetivo: corregir errores graves en `app.cpp`, reducir ruido de medicion y arreglar bugs funcionales visibles en CPU.

### Cambios en `app.cpp`

Problemas detectados:

- La ruta del dataset estaba hardcodeada a una maquina especifica.
- `delta` se creaba sin valor fijo, por lo que podia iniciar con basura o valores aleatorios.
- `TensorMath::sum` acumula sobre el tensor de salida, asi que `delta` debia limpiarse antes de cada reduccion.
- Se imprimia el vector completo de perdidas por batch, lo cual hacia el entrenamiento mucho mas lento.

Cambios:

- Se cambio la ruta del dataset a `datasets/mnist/flat`.
- `delta` ahora se inicializa con cero:

```cpp
Tensor delta = Tensor({1}, dev, 0.0f, true);
```

- Antes de calcular la perdida promedio se llama:

```cpp
delta.zeros();
```

- Se reemplazo la impresion completa de errores por:

```cpp
std::cout << "Loss: " << delta.data()[0] << std::endl;
```

### Cambios en `CpuMath`

Problemas detectados:

- `matmul` usaba triple loop generico con `int` y acceso poco favorable a cache.
- `im2col2d` y `col2im2d` usaban odometros con vectores y calculo de indices por coordenadas para cada elemento.
- `col2im2d` asumia que la salida ya estaba en cero antes de acumular.
- `_is_contiguous` tenia un loop peligroso con `size_t`.

Cambios:

- Se elimino `_is_contiguous` defectuoso.
- Se agrego un camino rapido para `matmul` 2D contiguo.
- Se reescribieron `im2col2d` y `col2im2d` con loops directos sobre `N, C, kernel_h, kernel_w, H_out, W_out`.
- `col2im2d` ahora limpia el tensor de salida antes de acumular.

### Bugs de pooling

Problemas detectados:

- `MaxPooling` no limpiaba toda la mascara antes del forward.
- `inv_max_pooling2D` podia dejar valores viejos en posiciones no visitadas.
- `inv_avg_pooling2D` incrementaba `patch_coords` con `out_shape._shapes` en vez de usar la forma de la ventana.
- Algunas operaciones con padding usaban `std::size_t`, lo cual puede causar underflow.

Cambios:

- La mascara de max pooling se limpia completa al iniciar.
- El backward de max pooling limpia toda la salida antes de escribir.
- El backward de average pooling usa `patch_shapes`.
- Se usaron indices firmados (`ptrdiff_t`) para coordenadas que pueden caer en padding.

### Tests

Se reemplazo `tests/pooling_test.cpp` por una prueba determinista de:

- max pooling forward;
- max pooling backward;
- average pooling forward;
- average pooling backward.

Validacion usada:

```powershell
cmake --build build --target pooling_test conv_test main --config Debug
.\build\Debug\pooling_test.exe
.\build\Debug\conv_test.exe
cmake --build build --target main --config Release
.\build\Release\main.exe
```

## Segundo paquete de mejoras

Objetivo: optimizar rutas calientes de BatchNorm y corregir representacion de vistas broadcast.

### Cambios en `Shape`

Problema detectado:

Las operaciones `unsqueeze`, `inner_pad` y `outer_pad` recomponian strides como si las dimensiones agregadas tuvieran memoria real. Para usos como BatchNorm, donde se expande un vector `(C)` a una vista tipo `(1, C, 1, 1)`, esas dimensiones deben comportarse como broadcast, con stride `0`.

Cambios:

- `unsqueeze` conserva los strides existentes y asigna stride `0` a la dimension agregada.
- `inner_pad` agrega dimensiones internas con stride `0`.
- `outer_pad` agrega dimensiones externas con stride `0`.

Esto hizo que tensores como `gamma`, `beta`, `mean` y `variance` puedan expandirse de forma segura para operar contra tensores `(N, C, H, W)`.

### Fast paths en `CpuMath`

Problema detectado:

BatchNorm usa muchas operaciones elementwise con broadcast por canal:

```text
(N, C, H, W) op (1, C, 1, 1)
```

La ruta generica usaba odometros y calculo de indices por elemento.

Cambios:

- Se agrego deteccion de tensores contiguos.
- Se agrego deteccion de broadcast por canal 4D.
- `add`, `sub`, `multiply` y `divide` ahora tienen caminos rapidos para:
  - tensores contiguos con misma forma;
  - broadcast por canal `(N, C, H, W)` contra `(1, C, 1, 1)`.

### Reducciones por canal

Problema detectado:

`mean`, `variance` y `reduce_all` usaban ruta generica, pero BatchNorm las llama todo el tiempo sobre el eje de canales.

Cambios:

- Se agregaron caminos directos para `axis == 1` y tensores 4D contiguos.
- Esto evita crear coordenadas por elemento y reduce trabajo de indice.

Validacion usada:

```powershell
cmake --build build --target tensor_test pooling_test conv_test conv_layer_test main --config Debug
.\build\Debug\tensor_test.exe
.\build\Debug\pooling_test.exe
.\build\Debug\conv_test.exe
.\build\Debug\conv_layer_test.exe
cmake --build build --target main --config Release
.\build\Release\main.exe
```

## Tercer paquete de mejoras

Objetivo: corregir problemas que podian hacer explotar el entrenamiento y reducir desperdicio de memoria.

### `TensorMath::sum`

Problema detectado:

`CpuMath::sum` acumula con:

```cpp
out[out_idx] += a[a_idx];
```

Eso es correcto a bajo nivel, pero `TensorMath::sum` exponia esa semantica sin limpiar la salida. Como consecuencia, cualquier llamada a `TensorMath::sum` podia sumar encima de resultados anteriores.

Caso importante:

- `FullyConnected::backward` usa `TensorMath::sum` para calcular `bias_grad`.
- Si `bias_grad` no se limpia antes, se acumulan gradientes entre batches.
- Esto puede aumentar las actualizaciones de bias y desestabilizar el entrenamiento.

Cambio:

`TensorMath::sum` ahora llama:

```cpp
out.zeros();
```

antes de delegar a `dev->math->sum(...)`.

### BatchNorm backward

Problema detectado:

La formula anterior de `BatchNorm::backward` no implementaba correctamente el gradiente respecto a la entrada. En particular, mezclaba terminos como si fueran multiplicaciones:

```text
dY * dBeta
xHat * gamma
```

cuando la formula compacta esperada es:

```text
dX = (gamma / sqrt(var + eps)) * (M*dY - dBeta - xHat*dGamma) / M
```

Cambios:

- Se calcula primero `M*dY - dBeta`.
- Luego se resta `xHat*dGamma`.
- Luego se divide por `sqrt(var + eps)`.
- Luego se multiplica por `gamma`.
- Finalmente se escala por `1/M`.

Tambien se reemplazaron operadores coma en las expansiones por sentencias separadas con `;`, para que el codigo sea claro y no parezca un typo.

### `OutputLayer`

Problema detectado:

`OutputLayer::forward` reservaba un tensor temporal `log_probs` en cada batch.

Cambio:

Se reutiliza `_backward_out` como buffer temporal durante forward de entrenamiento:

```cpp
__check_initialization(this->_backward_out, in.shape);
Tensor& log_probs = *(this->_backward_out);
```

Esto reduce asignaciones temporales por batch.

### Softmax 2D

Problema detectado:

La salida de la red usa softmax sobre tensores 2D contiguos `(batch, clases)`. La ruta generica era mas costosa de lo necesario.

Cambio:

Se agrego un camino rapido para `softmax` cuando la entrada y salida son tensores 2D contiguos.

### Test de BatchNorm

Se agrego `tests/batch_norm.cpp` y el target `batch_norm_test` en `CMakeLists.txt`.

La prueba valida una propiedad simple:

- Si `dY` es todo unos en BatchNorm de un canal, entonces `dX` debe ser practicamente cero.

Esto ayuda a detectar errores en la formula de backward.

Validacion usada:

```powershell
cmake --build build --target tensor_test pooling_test conv_test conv_layer_test cnn_test main --config Debug
.\build\Debug\tensor_test.exe
.\build\Debug\pooling_test.exe
.\build\Debug\conv_layer_test.exe
.\build\Debug\cnn_test.exe
cmake --build build --target batch_norm_test --config Debug
.\build\Debug\batch_norm_test.exe
cmake --build build --target main --config Release
.\build\Release\main.exe
```

## Cuarto paquete de mejoras

Objetivo: reducir el costo del experimento en `main` para depurar entrenamiento mas rapido.

Problema:

El entrenamiento en `main` recorria demasiadas muestras para una prueba rapida, y el batch era de 64.

Cambio:

Se paso a:

```cpp
std::size_t batch_size = 32;
std::size_t samples_per_epoch = 320;
```

La condicion del bucle ahora usa:

```cpp
while(curr_index + batch_size <= samples_per_epoch)
```

Con esto cada epoca de debug usa 10 batches: indices `0, 32, 64, ..., 288`.

Validacion usada:

```powershell
cmake --build build --target main --config Release
.\build\Release\main.exe
```

En una corrida corta se observo que la perdida bajaba rapidamente de valores alrededor de `3.x` hacia `2.x`, lo cual es una senal inicial mas saludable que una perdida explosiva.

## Archivos tocados

Archivos principales modificados:

- `project/app.cpp`
- `project/src/core/BatchNormLayer.cpp`
- `project/src/core/OutputLayer.cpp`
- `project/src/core/Shape.cpp`
- `project/src/math/CpuMath.cpp`
- `project/src/math/TensorMath.cpp`
- `tests/batch_norm.cpp`
- `tests/pooling_test.cpp`
- `CMakeLists.txt`

## Pendientes recomendados

- Cambiar la convencion de `update_weight` para que use descenso con `param -= lr * grad`, y asi poder usar `lr` positivo.
- Agregar gradient checking numerico para `BatchNorm`, `FullyConnected` y `Conv2D`.
- Medir con benchmarks en Release antes y despues de optimizaciones.
- Evitar mutar gradientes en sitio durante `update_weight`, porque puede dificultar debug.
- Reutilizar buffers en mas capas para reducir asignaciones por batch.
- Crear una opcion configurable en `app.cpp` para `epoch`, `batch_size`, `samples_per_epoch` y `lr`.
