#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>
#include <chrono>
#include <cmath>

#include "../lib/gbm.hpp"
#include "../lib/net.hpp"
#include "../lib/param.hpp"

int main(int argc, char *argv[])
{
    std::cout << std::fixed;
    std::cout.precision(12);

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

    std::vector<Net> ensemble(BATCH);
    for(unsigned int i = 0; i < BATCH; i++) {
        for(unsigned int l = 0; l < LAYERS; l++)
            ensemble[i].add_layer(EXT, EXT);
        ensemble[i].add_layer(EXT, OUT);
    }

    std::cout << "INITIALIZED NETWORK PARAMETERS!\n";
    net.model();

    for(unsigned int itr = 0; itr < ITR; itr++) {
        std::vector<std::thread> threads;
        for(unsigned int i = 0; i < BATCH; i++) {
            copy(net, ensemble[i], 1.00);
            threads.push_back(std::thread(&Net::train, std::ref(ensemble[i]),
                                          std::ref(x[i]), std::ref(y[i]), ALPHA, LAMBDA));
            if(itr == 0) {
                std::thread::id thread_id = threads[i].get_id();
                std::cout << "THREAD #" << thread_id << ": ENSEMBLE #" << i;
                std::cout << " (mu=" << y[i][MU] << ", sigma=" << y[i][SIGMA] << ")\n";
            }
        }
        if(itr == 0) std::cout << "\n";

        net.zero();
        for(unsigned int i = 0; i < BATCH; i++) {
            threads[i].join();
            add(ensemble[i], net, 1.00 / BATCH);
        }

        double loss = 0.00;
        for(unsigned int i = 0; i < BATCH; i++) {
            std::vector<double> out = net.forward(x[i]);
            for(unsigned int k = 0; k < OUT; k++)
                loss += pow(y[i][k] - out[k], 2);
        }
        loss /= BATCH;
        std::cout << "ITR #" << itr << " LOSS=" << loss << "\n";
    }

    net.model();

    return 0;
}