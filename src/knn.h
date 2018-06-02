#pragma once

#include <utility>
#include <algorithm>
#include "vector_ext.h"

using namespace std;

template <class T>
static int mode(const vector<pair<T, int>>& Z, const unsigned int k) {
    vector<int> freq;
    int mode_freq = 0;
    int mode = -1;

    auto it_end = Z.begin() + k;
    for (auto it = Z.begin(); it < it_end; it++) {
        size_t label = get<1>(*it);

        freq.reserve(label);
        if (freq.size() <= label) {
            freq.resize(label + 1);
        }
        freq[label]++;

        if (freq[label] > mode_freq) {
            mode_freq = freq[label];
            mode = label;
        }
    }

    return mode;
}

template <class T>
int knn(const vector<vector<T>>& X, const vector<int>& y, const vector<T>& z, unsigned long k) {
    vector<pair<T, int>> Z(y.size());

    // Calcula la distancia Euclideana entre z y todos las muestras del modelo
    for (size_t i = 0; i < y.size(); i++) {
        Z[i] = make_pair(norm2(z - X[i]), y[i]);
    }

    // Ordena el vector de menor a mayor para tener al principio de éste
    // las k muestras más cercanos a z.
    sort(Z.begin(), Z.end(), [](const pair<T, int>& a, const pair<T, int>& b) {
        return get<0>(a) < get<0>(b);
    });

    // Devuelve la moda entre esos k elementos
    return mode(Z, min(Z.size(), k));
}
