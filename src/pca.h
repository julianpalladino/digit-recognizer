#pragma once

#include <random>
#include <algorithm>
#include <iterator>
#include <functional>
#include <cassert>

#include "matrix.h"
#include "vector_ext.h"
#include "common.h"

using namespace std;

template <class T> Matrix<float> covariance_matrix(const vector<vector<T>>& X);

vector<float> random_vector(int size) {
    random_device rnd_device;
    mt19937 mersenne_engine(rnd_device());
    uniform_real_distribution<float> dist(0.0, 1.0);

    auto gen = bind(dist, mersenne_engine);
    vector<float> vec(size);
    generate(begin(vec), end(vec), gen);

    return vec;
}

// Calcula las componentes principales de la matriz de covarianzas de X, y devuelve
// los primeros +alpha+ autovectores y autovalores.
template <class T>
pair<vector<float>, Matrix<float>> pca(const vector<vector<T>>& X, int alpha, int n_iter) {
    auto Mx = covariance_matrix(X);
    //cerr << "Matriz de covarianza de X lista: tamaño=" << Mx.rows() << " x " << Mx.cols() << endl;

    alpha = min(Mx.rows(), alpha);  // para asegurarse de no irse de rango...

    vector<float> eigenvalues(alpha);
    Matrix<float> eigenvectors(Mx.rows(), alpha);

    // sacamos autovalores y autovectores de Mx
    for (int j = 0; j < alpha; j++) {
        auto randvec = random_vector(Mx.rows());
        pair<float, Matrix<float> > eigenv = power_method(Mx, randvec, n_iter); // ?

        float lambda = get<0>(eigenv);
        Matrix<float>& v = get<1>(eigenv);
        deflate(Mx, v, lambda);

        eigenvalues[j] = lambda;
        for (int i = 0; i < v.rows(); i++) {
            eigenvectors(i, j) = v(i, 0);
        }
        //cerr << "lambda_" << j << ": " << lambda << endl;
    }

    return make_pair(eigenvalues, eigenvectors);
}

template <class T>
Matrix<float> covariance_matrix(const vector<vector<T>>& X) {
    const int n = X.size();
    assert(n > 0);

    const int m = X[0].size();

    // Calcula la media para cada feature
    vector<float> mu = sum(X) / float(n);

    // Construye la matriz de covarianzas
    Matrix<float> Xc(n, m);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            Xc(i, j) = X[i][j] - mu[j];
        }
    }

    // Hace (Xc' * Xc) / (n - 1)
    auto Mx = Xc.mult_transposed();
    Mx = Mx / (n - 1);

    return Mx;
}

template <class T>
pair<T, Matrix<T>> power_method(const Matrix<T>& X, vector<T> base_vector, int n_iter) {
    Matrix<T> v(base_vector);
    Matrix<T> X_v(X.rows(), 1);
    T old_norm = 0;
    for (int i = 0; i < n_iter; i++) {
        X_v = X.mult(v);

        auto norm = X_v.norm_vect();
        assert(norm > 0);

        v = X_v / norm;

        if (abs(old_norm - norm) < 1e-10) {
            //cerr << "Converge en i = " << to_string(i) << ": " << to_string(norm) << endl;
            break;
        }
        old_norm = norm;
    }
    auto vt = v.transpose();
    auto vt_X_v = vt.mult(X_v);
    auto norm = v.norm_vect();
    return make_pair((vt_X_v(0,0) / norm), v);
}

/*
* Dada la matriz A, realiza un paso de deflación, es decir:
    A = A - lambda*V*V^t
*/
template <class T>
void deflate(Matrix<T>& A, const Matrix<T>& V, const T lambda) {
    auto B = V.transpose().mult_transposed();
    B = B * lambda;
    A = A - B;
}

// OJO! En el enunciado tc esta mal definido,
// usamos la definicion de la pag 22 de la presentacion del TP:
// sea x algún x^(i), tc(x) = V^t . x = ((v_1)^t .x, ..., (v_alpha)^t. x)^t, perteneciente a R mx1
// el alpha debe ser variable, para luego ser parametro de testeo
template <class T>
vector<float> tc(const Matrix<float>& V, const vector<T>& x, int alpha) {
    alpha = min(V.cols(), alpha); // para asegurarse de no irse de rango

    vector<float> res(alpha, 0);
    for (int j = 0; j < alpha; j++) {
        //calculo el elemento i, definido como (v_i)^t * x
        float suma = 0;
        for (size_t i = 0; i < x.size(); i++) {
            // (v_i) es la columna i de V
            suma = suma + V(i, j) * x[i];
        }
        res[j] = suma;
    }
    return res;
}
