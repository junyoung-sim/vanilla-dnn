#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>
#include <chrono>

#include "../lib/gbm.hpp"
#include "../lib/net.hpp"

#define BATCH 10
#define LAYERS 3
#define EXT 10
#define OUT 2

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

    Net net;
    for(unsigned int l = 0; l < LAYERS; l++)
        net.add_layer(EXT, EXT);
    net.add_layer(EXT, OUT);
    net.init(seed);

    std::vector<Net> ensemble(BATCH);
    for(unsigned int n = 0; n < BATCH; n++) {
        for(unsigned int l = 0; l < LAYERS; l++)
            ensemble[n].add_layer(EXT, EXT);
        ensemble[n].add_layer(EXT, OUT);
        copy(net, ensemble[n], 1.00);
    }

    return 0;
}