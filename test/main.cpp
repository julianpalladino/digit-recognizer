#include <cassert>
#include <iostream>
#include "matrix_tests.h"
#include "pca_tests.h"

using namespace std;

int main() {
    run_matrix_tests();
    run_pca_tests();

    cout << "Tests OK!" << endl;
    return 0;
}
