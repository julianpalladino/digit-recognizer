#pragma once

#include <cmath>
#include <vector>
#include <stdexcept>

using namespace std;

template <class T>
ostream& operator<<(ostream& out, const vector<T>& v) {
    for (size_t i = 0; i < v.size(); i++) {
        out << v[i] << " ";
    }
    out << "\n";
    return out;
}

template <class T>
vector<T> operator-(const vector<T> &a) {
    vector<T> res(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        res[i] = -a[i];
    }
    return res;
}

template <class T, class T2>
vector<T> operator+(const vector<T> &a, const vector<T2> &b) {
    if (a.size() != b.size()) {
        throw invalid_argument("Los tamaños no son iguales: " +
                to_string(a.size()) + " != " + to_string(b.size()));
    }
    vector<T> res(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        res[i] = a[i] + b[i];
    }
    return res;
}

template <class T, class T2>
vector<T> operator-(const vector<T> &a, const vector<T2> &b) {
    return a + (-b);
}

template <class T, class T2>
vector<T> operator*(const vector<T> &a, const T2 &k) {
    vector<T> res(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        res[i] = a[i] * k;
    }
    return res;
}

template <class T, class T2>
vector<T> operator*(const T2 &k, const vector<T> &a) {
    return a * k;
}

template <class T, class T2>
vector<float> operator/(const vector<T> &a, const T2 &k) {
    vector<float> res(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        res[i] = a[i] / k;
    }
    return res;
}

template <class T, class T2>
vector<float> operator/(const T2 &k, const vector<T> &a) {
    return a / k;
}

template <class T>
float norm2(const vector<T>& v) {
    T sum = T();
    for (size_t i = 0; i < v.size(); i++) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

template <class T>
T sum(const vector<T>& a) {
    T res;
    if (a.empty()) {
        throw invalid_argument("vector vacío");
    }
    res = a[0];
    for (size_t i = 1; i < a.size(); i++) {
        res = res + a[i];
    }
    return res;
}
