#pragma once
#include <cassert>
#include "../src/matrix.h"

using namespace std;

static void test_matrix() {
    Matrix<float> A({
        {0.403259,0.480808,0.778592},
        {0.0982272,0.163712,0.981606},
        {-0.0654826,0.180077,0.98147},
    });

    Matrix<float> B = A;

    assert(A == B);
}

static void test_mult() {
    Matrix<float> id({ {1,0,0}, {0,1,0}, {0,0,1} });
    assert(id.rows() == 3);
    assert(id.cols() == 3);

    Matrix<float> unos(3, 1, 0);
    unos(0, 0) = 1;
    unos(1, 0) = 1;
    unos(2, 0) = 1;

    assert(unos.eq_aprox(id.mult(unos)));

    Matrix<float> id_4_dimensiones({ {1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1} });
    assert(id_4_dimensiones.rows() == 4);
    assert(id_4_dimensiones.cols() == 4);

    Matrix<float> unos_4_dimensiones(4, 1, 0);
    unos_4_dimensiones(0, 0) = 1;
    unos_4_dimensiones(1, 0) = 1;
    unos_4_dimensiones(2, 0) = 1;
    unos_4_dimensiones(3, 0) = 1;

    assert(unos_4_dimensiones.eq_aprox(id_4_dimensiones.mult(unos_4_dimensiones)));
}

static void test_mult_transposed() {
    Matrix<int> M({ {8,1,6}, {3,5,7}, {4,9,2}, {-1,5,14}, {1,-5,0} });
    auto M_mt = M.mult_transposed();

    Matrix<int> expected_M_mt({ {91,49,63}, {49,157,129}, {63,129,285} });
    assert(M_mt == expected_M_mt);
}

void run_matrix_tests() {
    test_matrix();
    test_mult();
    test_mult_transposed();
}
