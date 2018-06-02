#include <iomanip>
#include <iostream>
#include <unistd.h>
#include <vector>
#include "common.h"
#include "knn.h"
#include "pca.h"

using namespace std;

const int DEFAULT_PCA = 1;
const int DEFAULT_K = 3;
const int DEFAULT_ALPHA = 37;
const int DEFAULT_N_ITER = 1000;

vector<int> parse_vector(const string& line) {
    vector<int> res;
    auto list = parse_list(line);
    res.reserve(list.size());
    for (auto it = list.cbegin(); it != list.cend(); it++) {
        res.push_back(*it);
    }
    return res;
}

void print_usage(char* argv[]) {
    cerr << "uso: " << argv[0] << " -m <method> -i <train_set> -q <test_set> -o <classif>\n" \
         << "\n" \
         << "Opciones:\n" \
         << "\t-m <method>: Método a utilizar: 0 = kNN, 1 = kNN+PCA (default: 1)\n" \
         << "\t-i <train_set>: CSV de entrenamiento. Cada línea es una\n" \
         << "\t    observación, cuyo primer valor es la etiqueta asociada, y el resto\n" \
         << "\t    son las características de la observación.\n" \
         << "\t-q <test_set>: CSV de observaciones a predecir.\n" \
         << "\t-o <classif>: CSV con el resultado de la predicción de cada imagen.\n" \
         << "\n" \
         << "Parámetros:\n" \
         << "\tK:\tDefine los k vecinos más cercanos (default: " << DEFAULT_K << ")\n" \
         << "\tALPHA:\tCantidad de componentes principales a tomar \n" \
         << "\t\t(default: " << DEFAULT_ALPHA << ")\n" \
         << "\tN_ITER:\tCantidad de iteraciones del método de potencia \n" \
         << "\t\t(default: " << DEFAULT_N_ITER << ")\n" \
         << "\n" \
         << "Ejemplo:\n\tK=4 ALPHA=20 " << argv[0] << " -m 1 -i data/train.csv -q data/test.csv -o preds.csv\n" \
         << endl;
}

int main(int argc, char* argv[]) {
    if (argc == 1) {
        print_usage(argv);
        return 1;
    }

    bool use_pca = DEFAULT_PCA;
    string train_set_path, test_set_path, classif_path;

    int opt;
    while ((opt = getopt(argc, argv, "m:i:q:o:h")) != EOF) {
        switch(opt) {
            case 'm': {
                int method = atoi(optarg);
                if (method > 1 || method < 0) {
                    cerr << argv[0] << ": opción inválida -- 'm'. Valores posibles: 0, 1" << endl;
                    return 1;
                }
                use_pca = method == 1;
                break;
            }
            case 'i':
                train_set_path = optarg;
                break;
            case 'q':
                test_set_path = optarg;
                break;
            case 'o':
                classif_path = optarg;
                break;
            case 'h':
                print_usage(argv);
                return 1;
        }
    }

    if (use_pca) {
        cerr << "Método: k-NN + PCA" << endl;
    } else {
        cerr << "Método: k-NN" << endl;
    }

    // Setea el parámetor k del KNN
    int k_knn = DEFAULT_K;
    if (!get_env_var("K").empty()) {
        k_knn = stoi(get_env_var("K"));
    }
    cerr << "Parámetro K = " << k_knn << endl;

    // Setea el parámetro de corte Alpha para PCA
    int alpha = DEFAULT_ALPHA;
    if (!get_env_var("ALPHA").empty()) {
        alpha = stoi(get_env_var("ALPHA"));
    }
    cerr << "Parámetro ALPHA = " << alpha << endl;

    // Setea el parámetro de iteraciones del método de la potencia
    int n_iter = DEFAULT_N_ITER;
    if (!get_env_var("N_ITER").empty()) {
        n_iter = stoi(get_env_var("N_ITER"));
    }
    cerr << "Parámetro N_ITER = " << n_iter << endl;

    ifstream train_set_f(train_set_path);

    if (!train_set_f.is_open()) {
        cerr << "No se pudo abrir " << train_set_path << "." << endl;
        return 1;
    }

    vector<vector<int>> X;
    vector<int> y;

    // Lee el dataset de entrenamiento
    cerr << "Leyendo dataset de " << train_set_path << "... ";
    read_dataset_from_stream(train_set_f, X, y);
    cerr << "listo." << endl;
    train_set_f.close();
    cerr << "Tamaño dataset: " << X.size() << endl;

    if (X.empty()) {
        cerr << "El dataset está vacío" << endl;
        return 1;
    }

    const size_t num_features = X[0].size();

    ifstream test_set_f(test_set_path);
    ofstream classif_f(classif_path);

    // Si el test set está etiquetado, se van a contar todos los true positives
    // y calcular accuracy al final de todo.
    bool must_evaluate = false;
    int tps = 0;
    int total = 0;

    classif_f << "ImageId,Label" << endl;

    if (use_pca) {
        cerr << "Corriendo PCA... ";
        pair<vector<float>, Matrix<float>> pca_res = pca(X, alpha, n_iter);
        cerr << "listo." << endl;

        vector<float>& lambdas = get<0>(pca_res);
        Matrix<float>& V = get<1>(pca_res);

        cerr << "Autovalores: " << lambdas;

        // Aplica la transformación caracterísitca a todo X
        cerr << "Transformando dataset... ";
        vector<vector<float>> X_;
        X_.reserve(X.size());
        for (size_t i = 0; i < X.size(); i++) {
            X_.push_back(tc(V, X[i], alpha));
        }
        cerr << "listo." << endl;

        string line;
        getline(test_set_f, line);   // Saltea primer linea (headers)

        while (getline(test_set_f, line)) {
            int true_label = -1;
            auto z = parse_vector(line);
            if (z.size() == num_features + 1) {
                must_evaluate = true;
                true_label = z[0];
                vector<int>(z.begin()+1, z.end()).swap(z);
            }

            auto z_ = tc(V, z, alpha);
            auto pred_label = knn(X_, y, z_, k_knn);
            classif_f << total + 1 << "," << pred_label << "\n";

            if (true_label >= 0 && true_label == pred_label) tps++;
            total++;
        }
    } else {
        string line;
        getline(test_set_f, line);   // Saltea primer linea (headers)

        while (getline(test_set_f, line)) {
            int true_label = -1;
            auto z = parse_vector(line);
            if (z.size() == num_features + 1) {
                must_evaluate = true;
                true_label = z[0];
                vector<int>(z.begin()+1, z.end()).swap(z);
            }

            auto pred_label = knn(X, y, z, k_knn);
            classif_f << total + 1 << "," << pred_label << "\n";

            if (true_label >= 0 && true_label == pred_label) tps++;
            total++;
        }
    }

    classif_f << flush;

    if (must_evaluate) {
        cerr << "Total: " << total << endl;
        cerr << "True positives: " << tps << endl;
        cerr << "Accuracy: " << setprecision(5) << tps / (double)total << endl;
    }

    test_set_f.close();
    classif_f.close();

    return 0;
}
