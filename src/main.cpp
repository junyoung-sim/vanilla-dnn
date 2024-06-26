#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>
#include <chrono>

#include "../lib/gbm.hpp"
#include "../lib/net.hpp"

#define BATCH 10
#define LAYERS 100
#define EXT 100
#define OUT 2

#define ALPHA 0.000001
#define LAMBDA 0.10

int main(int argc, char *argv[])
{
    std::cout << std::fixed;
    std::cout.precision(6);

    std::normal_distribution<double> gaussian(0.00, 0.01);
    std::default_random_engine seed(std::chrono::system_clock::now().time_since_epoch().count());

    std::vector<GBMParam> param(BATCH);
    std::vector<std::vector<double>> y(BATCH, std::vector<double>(OUT));
    for(unsigned int i = 0; i < BATCH; i++) {
        double mu = gaussian(seed);
        double sigma = gaussian(seed);
        param[i] = GBMParam(1.00, mu, sigma);
        y[i] = {mu, sigma};
    }
    std::vector<std::vector<double>> x = gbm(param, EXT, seed);
    std::cout << "GENERATED TEST DATASET!\n\n";

    Net net;
    for(unsigned int l = 0; l < LAYERS; l++)
        net.add_layer(EXT, EXT);
    net.add_layer(EXT, OUT);
    net.init(seed);
    std::cout << "INITIALIZED NETWORK PARAMETERS!\n";

    net.model();

    std::vector<Net> ensemble(BATCH);
    for(unsigned int i = 0; i < BATCH; i++) {
        for(unsigned int l = 0; l < LAYERS; l++)
            ensemble[i].add_layer(EXT, EXT);
        ensemble[i].add_layer(EXT, OUT);
        copy(net, ensemble[i], 1.00);
    }
    std::cout << "COPIED NETWORK PARAMETERS TO BATCH ENSEMBLE!\n\n";

    std::vector<std::thread> threads;
    for(unsigned int i = 0; i < BATCH; i++)
        threads.push_back(std::thread(&Net::train, ensemble[i], std::ref(x[i]), std::ref(y[i]), ALPHA, LAMBDA));
    for(unsigned int i = 0; i < BATCH; i++)
        threads[i].join();

    return 0;
}