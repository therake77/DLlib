
## Tensor
Un tensor, para fines de este proyecto, es un array multidimensional.
Cada tensor tiene:
- Un puntero a sus datos en memoria (`data`)
- Una forma (`shape`) que representa sus dimensiones
- Un método especial `to`, que permite enviar - obtener sus datos de CPU - GPU
- Un objeto `Device` que permite la abstracción de la capa de hardware

## Hardware
Los objetos `Tensor` también encapsulan sus datos en GPU a través de un **mecanismo de abstracción de hardware**, encargado por la clase base `Device` y la clase Factory `DeviceManager`.  
`Device` sirve como clase base para `CpuDevice` y `GpuDevice`, los cuales implementan los mismos métodos de la clase base, sin embargo en distintos dispositivos. Esta capa implementa:
- Manejo de memoria (asignamiento, liberación)
- Operaciones primitivas sobre memoria ( copiado, guardado, relleno con un valor, etc )

