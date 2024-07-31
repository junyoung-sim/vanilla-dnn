#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>
#include <chrono>
#include <cmath>

#include "../lib/gbm.hpp"
#include "../lib/net.hpp"

#define MU 0
#define SIGMA 1
#define N 10000
#define BATCH 10
#define LAYERS 100
#define EXT 100
#define OUT 2
#define ALPHA 0.000001
#define LAMBDA 0.10
#define ITR 100

std::normal_distribution<double> gaussian(0.00, 0.01);
std::default_random_engine seed(std::chrono::system_clock::now().time_since_epoch().count());

std::vector<GBMParam> param;
std::vector<std::vector<double>> x;
std::vector<std::vector<double>> y;

Net net;
std::vector<Net> ensemble;

void generate_dataset() {
    param.resize(N);
    y.resize(N, std::vector<double>(OUT));
    for(unsigned int i = 0; i < N; i++) {
        double mu = gaussian(seed);
        double sigma = gaussian(seed);
        param[i] = GBMParam(1.00, mu, sigma);
        y[i] = {mu, sigma};
    }
    x = gbm(param, EXT, seed);
    std::cout << "GENERATED TEST DATASET!\n\n";
}

void initialize() {
    for(unsigned int l = 0; l < LAYERS; l++)
        net.add_layer(EXT, EXT);
    net.add_layer(EXT, OUT);
    net.init(seed);

    ensemble.resize(BATCH);
    for(unsigned int i = 0; i < BATCH; i++) {
        for(unsigned int l = 0; l < LAYERS; l++)
            ensemble[i].add_layer(EXT, EXT);
        ensemble[i].add_layer(EXT, OUT);
    }

    std::cout << "INITIALIZED NETWORK PARAMETERS!\n";
    net.model();
}

int main(int argc, char *argv[])
{
    std::cout << std::fixed;
    std::cout.precision(12);

    generate_dataset();
    initialize();

    for(unsigned int itr = 0; itr < ITR; itr++) {
        std::vector<std::thread> threads;
        std::vector<unsigned int> index(N);
        std::iota(index.begin(), index.end(), 0);
        std::shuffle(index.begin(), index.end(), seed);

        for(unsigned int i = 0; i < BATCH; i++) {
            unsigned int k = index[i];
            copy(net, ensemble[i], 1.00);
            threads.push_back(std::thread(&Net::train, std::ref(ensemble[i]),
                                          std::ref(x[k]), std::ref(y[k]), ALPHA, LAMBDA));
            //std::thread::id thread_id = threads[i].get_id();
            //std::cout << "THREAD #" << thread_id << ": ENSEMBLE #" << i;
            //std::cout << " (mu=" << y[k][MU] << ", sigma=" << y[k][SIGMA] << ")\n";
        }

        net.zero();
        for(unsigned int i = 0; i < BATCH; i++) {
            threads[i].join();
            add(ensemble[i], net, 1.00 / BATCH);
        }
        
        double loss = 0.00;
        for(unsigned int i = 0; i < BATCH; i++) {
            std::vector<double> out = net.forward(x[i]);
            for(unsigned int j = 0; j < OUT; j++)
                loss += pow(y[i][j] - out[j], 2);
        }
        loss /= BATCH;
        std::cout << "ITR #" << itr << " LOSS=" << loss << "\n";
        //std::cout << "\n";
    }

    net.model();

    return 0;
}