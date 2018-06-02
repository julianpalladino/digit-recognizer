#include <iostream>
#include <utility>
#include <list>
#include <vector>
#include "../src/pca.h"
#include "../src/common.h"

using namespace std;

const int DEFAULT_N_ITER = 1000;
const int DEFAULT_EPS = 10;

struct Parameters {
    int k_knn;
    int alpha;
    int k_kfold;
};

Parameters parse_header(const string& line) {
    string token;
    stringstream ss(line);

    getline(ss, token, ' ');
    // ignora primer token...

    getline(ss, token, ' ');
    int k_knn = stoi(token);
    getline(ss, token, ' ');
    int alpha = stoi(token);
    getline(ss, token, ' ');
    int k_kfold = stoi(token);

    return Parameters { k_knn, alpha, k_kfold };
}

pair<Parameters, list<vector<vector<int>>>> read_input_file(istream& f, const vector<vector<int>>& X) {
    // Primero lee los parametros de la primer línea
    string line;
    getline(f, line);
    auto params = parse_header(line);

    list<vector<vector<int>>> partitions;

    // Ahora lee cada línea para saber cómo está particionado el training dataset
    for (int k = 0; k < params.k_kfold; k++) {
        vector<vector<int>> Xk;
        getline(f, line);

        string token;
        stringstream ss(line);

        int i = 0;
        while (getline(ss, token, ' ')) {
            if (token == "1") {
                Xk.push_back(X[i]);
            }
            i++;
        }

        partitions.push_back(Xk);
    }

    return make_pair(params, partitions);
}

list<vector<float>> read_expected_file(istream& f, const int alpha, const int k_kfold) {
    list<vector<float>> res;

    // Ahora lee cada línea para saber cómo está particionado el training dataset
    string line;
    for (int k = 0; k < k_kfold; k++) {
        vector<float> pca;
        for (int i = 0; i < alpha; i++) {
            getline(f, line);
            pca.push_back(stod(line));
        }
        res.push_back(pca);
    }

    return res;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "uso: " << argv[0] << " TRAIN_SET TEST.IN TEST.EXPECTED" << endl;
        return 1;
    }

    const string trainset_path(argv[1]);
    const string in_path(argv[2]);
    const string expected_path(argv[3]);

    ifstream trainset_f(trainset_path);
    if (!trainset_f.is_open()) {
        cerr << "No se pudo abrir " << trainset_path << "." << endl;
        return 1;
    }

    ifstream in_f(in_path);
    if (!in_f.is_open()) {
        cerr << "No se pudo abrir " << in_path << "." << endl;
        return 1;
    }

    ifstream expected_f(expected_path);
    if (!expected_f.is_open()) {
        cerr << "No se pudo abrir " << expected_path << "." << endl;
        return 1;
    }

    // Primero lee el dataset de entrenamiento.
    // X e y almacenan las muestras y sus etiquetas del dataset de
    // entrenamiento.
    vector<vector<int>> X;
    vector<int> y;

    cerr << "Test de integración: " << trainset_path << ", " << in_path << ", "
         << expected_path << endl;

    read_dataset_from_stream(trainset_f, X, y);
    trainset_f.close();

    // Ahora lee los parámetros de test.in, y crea las particiones
    auto in_res = read_input_file(in_f, X);
    in_f.close();

    auto& params = get<0>(in_res);
    cerr << "Params: k_knn=" << params.k_knn
         << ", alpha=" << params.alpha
         << ", k_kfold=" << params.k_kfold
         << endl;

    auto pca_per_fold = read_expected_file(expected_f, params.alpha, params.k_kfold);

    auto& partitions = get<1>(in_res);
    auto it_part = partitions.cbegin();
    auto it_pca  = pca_per_fold.cbegin();

    int k = 0;
    for (; it_part != partitions.cend(); it_part++, it_pca++) {
        cerr << "[partición " << k << "] Corriendo PCA..." << endl;
        auto pca_res = pca(*it_part, params.alpha, DEFAULT_N_ITER);
        auto eigenvalues = get<0>(pca_res);
        cerr << "[partición " << k << "] PCA listo" << endl;

        assert(it_pca->size() == eigenvalues.size());

        bool different = false;
        for (size_t i = 0; i < it_pca->size(); i++) {
            if (abs((*it_pca)[i] - eigenvalues[i]) > DEFAULT_EPS) {
                cerr << "[partición " << k << "] Valor " << i << " da muy distinto: "
                     << (*it_pca)[i] << " !=~ " << eigenvalues[i] << endl;
                different = true;
            }
        }

        if (!different) {
            cerr << "[partición " << k << "] Los valores son iguales!" << endl;
        }

        k++;
    }

    // Libera recursos
    expected_f.close();

    return 0;
}
