#pragma once
//#include <cassert>
#include <cmath>
#include <iostream>
#include <tuple>
#include <vector>
#include "common.h"

using namespace std;

template <class T>
class Matrix {
    int rows_;
    int cols_;
    vector<T> M_;

  public:
    Matrix(int rows, int cols, T def = T()) :
        rows_(rows),
        cols_(cols),
        M_(rows * cols, def) { };

    Matrix(const Matrix<T>& rhs) :
        rows_(rhs.rows_),
        cols_(rhs.cols_),
        M_(rhs.M_) { };

    Matrix(const vector<T>& rhs) :
        rows_(rhs.size()),
        cols_(1),
        M_(rhs) { };

    Matrix(const vector<vector<T>>& rhs) :
        rows_(rhs.size()),
        cols_(rhs[0].size()) {
            M_.resize(rows_ * cols_);
            for (int i = 0; i < rows_; i++) {
                for (int j = 0; j < cols_; j++) {
                    assert(j < (int) rhs[i].size());
                    M_[i * cols_ + j] = rhs[i][j];
                }
            }
        };

    bool operator==(const Matrix<T>& rhs) const;

    int rows()   const { return rows_; };
    int cols()   const { return cols_; };
    int height() const { return rows_; };
    int width()  const { return cols_; };

    T& operator()(int i, int j);
    const T operator()(int i, int j) const;
    const T at(int i, int j) const;

    Matrix<T> operator-() const;
    Matrix<T> operator+(const Matrix<T> &rhs) const;
    Matrix<T> operator-(const Matrix<T> &rhs) const;
    Matrix<T> operator*(const T &rhs) const;
    Matrix<T> operator/(const T &rhs) const;
    template <typename U> friend Matrix<U> operator*(const U &lhs, const Matrix<U> &rhs);
    template <typename U> friend Matrix<U> operator/(const U &lhs, const Matrix<U> &rhs);

    Matrix<T> mult(const Matrix<T>& B) const;
    Matrix<T> transpose() const;
    Matrix<T> mult_transposed() const;
    T norm_vect() const;

    template <typename U>
    friend ostream& operator<<(ostream& stream, const Matrix<U>& M);

    bool eq_aprox(const Matrix<T>& B) const;

};

template <class T>
Matrix<T> Matrix<T>::transpose() const{
    Matrix<T> res = Matrix<T>(cols_, rows_, 0);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            res(j, i) = this->at(i, j);
        }
    }
    return res;
}

template<class T>
bool Matrix<T>::operator==(const Matrix<T>& rhs) const {
    return rows_ == rhs.rows_ && \
        cols_ == rhs.cols_ && \
        M_ == rhs.M_;
}

template <class T>
bool Matrix<T>::eq_aprox(const Matrix<T>& B) const {
    if (rows_ != B.rows() || cols_ != B.cols()) return false;
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            if (abs(this->at(i, j) - B(i, j)) >= eps) { // ¬(|a-b|<epsilon)
                return false;
            }
        }
    }
    return true;
}

template <class T>
T& Matrix<T>::operator()(int i, int j) {
    //assert(i >= 0 && i < rows_);
    //assert(j >= 0 && j < cols_);
    return M_[i * cols_ + j];
}

template <class T>
const T Matrix<T>::operator()(int i, int j) const {
    //assert(i >= 0 && i < rows_);
    //assert(j >= 0 && j < cols_);
    return M_[i * cols_ + j];
}

template <class T>
const T Matrix<T>::at(int i, int j) const {
    //assert(i >= 0 && i < rows_);
    //assert(j >= 0 && j < cols_);
    return M_[i * cols_ + j];
}

template <class T>
Matrix<T> Matrix<T>::operator-() const {
    Matrix<T> res(rows_, cols_);
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            res(i, j) = -this->at(i, j);
        }
    }
    return res;
}

template <class T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &rhs) const {
    if (rows_ != rhs.rows() || cols_ != rhs.cols()) {
        throw invalid_argument("Los tamaños no son iguales: " + to_string(rows_) + \
                "x" + to_string(cols_) + " != " + to_string(rhs.rows()) + "x" + \
                to_string(rhs.cols()));
    }
    Matrix<T> res(rows_, cols_);
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            res(i, j) = this->at(i, j) + rhs(i, j);
        }
    }
    return res;
}

template <class T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &rhs) const {
    return (*this) + (-rhs);
}

template <class T>
Matrix<T> Matrix<T>::operator*(const T &rhs) const {
    Matrix<T> res(rows_, cols_);
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            res(i, j) = this->at(i, j) * rhs;
        }
    }
    return res;
}

template <class T>
Matrix<T> operator*(const T &lhs, const Matrix<T>& rhs) {
    return rhs * lhs;
}

template <class T>
Matrix<T> Matrix<T>::operator/(const T &rhs) const {
    Matrix<T> res(rows_, cols_);
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            res(i, j) = this->at(i, j) / rhs;
        }
    }
    return res;
}

template <class T>
Matrix<T> operator/(const T &lhs, const Matrix<T>& rhs) {
    return rhs / lhs;
}

template <class T>
Matrix<T> Matrix<T>::mult(const Matrix<T>& B) const {
    if (cols_ != B.rows()) {
        throw runtime_error("No se puede multiplicar matriz de " \
            "(" + to_string(rows_) + ", " + to_string(cols_) + ") con " \
            "(" + to_string(B.rows()) + ", " + to_string(B.cols()) + ")");
    }

    Matrix<T> res(rows_, B.cols(), 0);

    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < B.cols(); j++) {
            res(i, j) = T();
            for (int k = 0; k < B.height(); k++) {
                res(i, j) += this->at(i, k) * B(k, j);
            }
        }
    }

    return res;
}

// Multiplica la matriz con su transpuesta, por izquierda: A' * A
// Considerando que (A * A')' = A'' * A' = A * A', también sabemos que la
// matriz resultante es simetrica.
template <class T>
Matrix<T> Matrix<T>::mult_transposed() const {
    Matrix<T> res(cols_, cols_, 0);

    for (int k = 0; k < rows_; k++) {
        for (int i = 0; i < cols_; i++) {
            for (int j = i; j < cols_; j++) {
                res(i, j) += this->at(k, i) * this->at(k, j);
            }
        }
    }

    // Por simetría, copio los valores
    for (int i = 0; i < cols_; i++) {
        for (int j = 0; j < cols_; j++) {
            res(j, i) = res(i, j);
        }
    }

    return res;
}

template <class T>
ostream& operator<<(ostream& out, const Matrix<T>& rhs) {
    for (int i = 0; i < rhs.rows_; i++) {
        for (int j = 0; j < rhs.cols_; j++) {
            out << rhs(i, j) << "\t";
        }
        out << "\n";
    }
    return out;
}

template <class T>
T Matrix<T>::norm_vect() const {
    T acum = 0;
    for (int i = 0; i < rows_; i++) {
        auto vi = this->at(i, 0);
        acum += vi * vi;
    }
    return sqrt(acum);
}
