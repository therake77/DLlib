## Im2Col
`im2col` (*image to column*) es una técnica usada en la convolución 2D (extendible a otras dimensiones) que simplifica la operación a una multiplicación de matrices.  

### Caso 2D
Si imaginamos al tensor de entrada (la imagen) como una matriz y al kernel de la misma forma, la idea es simple:
- Calcular cada posible ventana de la matriz de entrada $I$ sobre la cual el kernel $K$ se deslizará.
- Guardar estas ventanas como una fila (la matriz colapsa en una fila) en una nueva matriz grande, de tal forma que se van apilando de manera vertical.
- Crear una matriz formado por los *kernels* colapsados a una fila, en donde cada fila que representa a un *kernel* forma una columna.
Con esta idea, la convolución se reduce a una multiplicación de matrices 2D entre estas dos matrices construidas.

Si añadiéramos una dimensión más (canales):
- Cada canal del kernel se va simplemente acumulando a lo largo de la tercera dimensión, como filas.
- Lo mismo ocurre con los canales de la imagen: El tensor es tajado por canal, y el algoritmo anterior aplica. Las filas procesadas se van acumulando sobre la tercera dimensión.

Si se añadiera una cuarta dimensión, simplemente el procedimiento anterior se repite, es decir, las salidas 3D se van acumulando sobre la cuarta dimensión, apilándose.


### Caso ND

Una generalización simple (pero poco eficiente pues la cantidad de memoria redundante escalaría muy rápido), para cualquier convolución del tipo `convND` sería:
- Calcular cada "ventana" (ya no sería esta 2D) y transformarla en una fila, apilándolas en una matriz
- Cada kernel se convierte en una columna de otra matriz.
- Al final del procedimiento, multiplicar ambas matrices, y reconstruir las dimensiones del tensor de salida como último paso


