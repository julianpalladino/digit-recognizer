#pragma once
#include <cassert>
#include <iostream>
#include "../src/pca.h"

void test_power() {
    Matrix<float> M ({ {3,0,0}, {0,2,0}, {0,0,1} });

    vector<float> v (3, 1);
    pair<float, Matrix<float> > res = power_method(M, v, 30);

    float eigenval = get<0>(res);
    assert(abs(eigenval - 3) <= eps);

    Matrix<float> M2 ({ {7,2,3}, {0,2,0}, {-6,-2,-2} });

    vector<float> v2 (3, 1);
    v2[2] = 2;
    res = power_method(M2, v2, 30);
    eigenval = get<0>(res);

    assert(abs(eigenval - 4) <= eps);
}

void test_deflate() {
    Matrix<float> M ({ {3,0,0}, {0,2,0}, {0,0,1} });

    vector<float> v (3, 1);
    for (int i = 0; i < 3; i++) {
        pair<float, Matrix<float> > res = power_method(M, v, 30);
        float eigenval = get<0>(res);
        deflate(M, get<1>(res), get<0>(res));
        switch(i) {
            case 0: assert(abs(eigenval - 3) <= eps);
                    break;
            case 1: assert(abs(eigenval - 2) <= eps);
                    break;  
            case 2: assert(abs(eigenval - 1) <= eps);  
                    break;
        }
    }

    Matrix<float> M2 ({ {7,2,3}, {0,2,0}, {-6,-2,-2} });

    vector<float> v2 (3, 1);
    v2[2] = 2;

    for (int i = 0; i < 3; i++) {
        pair<float, Matrix<float> > res2 = power_method(M2, v2, 30);
        float eigenval2 = get<0>(res2);
        deflate(M2, get<1>(res2), get<0>(res2));
        switch(i) {
            case 0: assert(abs(eigenval2 - 4) <= eps);
                    break;
            case 1: assert(abs(eigenval2 - 2) <= eps);
                    break;  
            case 2: assert(abs(eigenval2 - 1) <= eps);  
                    break;
        }
    }
}

void test_tc() {
    Matrix<float> V({{1,2,3}, {4,5,6}, {7,8,9}});
    vector<float> x({1, 2, 3});

    //alpha = 3
    vector<float> res_correcto_alpha3({30, 36, 42});
    vector<float> res_alpha3 = tc(V, x, 3);
    assert(res_alpha3 == res_correcto_alpha3);

    //alpha = 2
    vector<float> res_correcto_alpha2({30, 36});
    vector<float> res_alpha2 = tc(V, x, 2);
    assert(res_alpha2 == res_correcto_alpha2);

    //alpha = 1
    vector<float> res_correcto_alpha1({30});
    vector<float> res_alpha1 = tc(V, x, 1);
    assert(res_alpha1 == res_correcto_alpha1);
}

void run_pca_tests() {
    test_power();
    test_deflate();
    test_tc();
}
