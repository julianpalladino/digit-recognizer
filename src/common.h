#pragma once

#include <string>
#include <vector>
#include <list>
#include <sstream>
#include <fstream>
#include <utility>
#include <stdexcept>

using namespace std;

const double eps = 1e-5;

list<int> parse_list(const string& line) {
    list<int> res;
    string token;
    stringstream ss(line);
    while (getline(ss, token, ',')) {
        if (!token.empty()) {
            res.push_back(stoi(token));
        }
    }
    return res;
}

size_t read_dataset_from_stream(istream& f,
        vector<vector<int>>& examples, vector<int>& labels)
{
    string line;
    list<list<int>> values;

    // Saltea primer línea porque son los headers del CSV
    getline(f, line);

    int i = 0;
    while (getline(f, line)) {
        try {
            values.push_back(parse_list(line));
        } catch (const std::invalid_argument &e) {
            cerr << "La línea " << i << " es inválida y fue ignorada: " << line << endl;
        }
        i++;
    }

    examples.reserve(values.size());
    labels.reserve(values.size());

    for (auto it_val = values.begin(); it_val != values.end(); it_val++) {
        // El primer entero de cada línea es la "etiqueta", y los enteros
        // que le siguen representan el valor del pixel de cada imagen.
        labels.push_back(*it_val->begin());

        // Agrega el resto como muestra
        vector<int> ex;
        ex.reserve(it_val->size());
        for (auto it = ++it_val->begin(); it != it_val->end(); it++) {
            ex.push_back(*it);
        }
        examples.push_back(ex);
    }

    return values.size();
}

string get_env_var(const string& key) {
    char* val = getenv(key.c_str());
    return val == NULL ? string("") : string(val);
}
