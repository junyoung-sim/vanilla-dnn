#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>
#include <chrono>

#include "../lib/net.hpp"

int main(int argc, char *argv[])
{
    std::cout << std::fixed;
    std::cout.precision(6);

    std::normal_distribution<double> gaussian(0.00, 1.00);
    std::default_random_engine seed(std::chrono::system_clock::now().time_since_epoch().count());

    // vanilla NN w/ 1M+ parameters
    Net net;
    for(unsigned int l = 0; l < 100; l++)
        net.add_layer(100, 100);
    net.add_layer(100, 10);
    net.init(seed);
    std::cout << "INITIALIZED NETWORK PARAMETERS.\n";

   

    return 0;
}