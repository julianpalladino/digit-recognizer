# OCR Digit Recognizer

## Documentation
* All documentation can be found in **Documentation - Informe.pdf**. This includes development and experimentation


## Dependencies

* cmake 3.5+
* g++

## Repository Structure

* `src/`:  Has the main source code
* `test/`: Unit tests
* `py/`: Python code, includes KnnClassifier which is a *wrapper* of `tp2`
* `doc/`: IPython Notebooks, experimentation code

## Use

###### Compilation of`tp2`

```
cmake .
make
```

###### To run unit tests and integration tests (PCA testing):

```
make check
```
###### Execution of `tp2`
```
$ ./tp2 
$ ./tp2 -m <method> -i <train_set> -q <test_set> -o <classif>
```
###### Options :
	-m <method>: Defines the method to use: 0 = kNN, 1 = kNN+PCA (default: 1)
	-i <train_set>: Training CSV. Each line is an
	    observation, in which the first value is its label, and the rest
	    are the characteristics.
	-q <test_set>: Test CSV. Observations to predict.
	-o <classif>: CSV with the prediction's result, corresponding with each image.

###### Parameters:
	K:	Defines the k of kNN (default: 3)
	ALPHA:	Amount of principal components taken in PCA
		(default: 37)
	N_ITER:	Amount of iterations done in power method
		(default: 1000)

###### Example:
	K=4 ALPHA=20 ./tp2 -m 1 -i data/train.csv -q data/test.csv -o preds.csv

