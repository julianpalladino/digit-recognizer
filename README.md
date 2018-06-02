# Métodos Numéricos TP2

## Dependencias

* cmake 3.5+
* g++

## Estructura de directorio

* `src/`:  Contiene el codigo fuente principal
* `test/`: Tests de unidad
* `py/`: Código de Python, incluye la clase KnnClassifier que actua de *wrapper* de `tp2`
* `doc/`: Notebooks de IPython, código de los experimentos

## Uso

Para compilar el ejecutable `tp2`

```
cmake .
make
```

Para correr los tests de unidad y de integración (chequeo de PCA):

```
make check
```

Se utiliza de la manera requerida en el enunciado del trabajo práctico.  Si se
corre con `-h` o sin parámetros imprime por pantalla los parámetros y opciones.

```
$ ./tp2 
uso: ./tp2 -m <method> -i <train_set> -q <test_set> -o <classif>

Opciones:
	-m <method>: Método a utilizar: 0 = kNN, 1 = kNN+PCA (default: 1)
	-i <train_set>: CSV de entrenamiento. Cada línea es una
	    observación, cuyo primer valor es la etiqueta asociada, y el resto
	    son las características de la observación.
	-q <test_set>: CSV de observaciones a predecir.
	-o <classif>: CSV con el resultado de la predicción de cada imagen.

Parámetros:
	K:	Define los k vecinos más cercanos (default: 3)
	ALPHA:	Cantidad de componentes principales a tomar
		(default: 37)
	N_ITER:	Cantidad de iteraciones del método de potencia
		(default: 1000)

Ejemplo:
	K=4 ALPHA=20 ./tp2 -m 1 -i data/train.csv -q data/test.csv -o preds.csv
```
