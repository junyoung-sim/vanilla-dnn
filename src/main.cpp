#include <cstdlib>
#include <iostream>
#include <random>
#include <chrono>

#include "../lib/net.hpp"

int main(int argc, char *argv[])
{
    std::cout << std::fixed;
    std::cout.precision(6);

    std::default_random_engine seed(std::chrono::system_clock::now().time_since_epoch().count());

    Net net;
    net.add_layer(10, 10);
    net.add_layer(10, 10);
    net.add_layer(10, 5);
    net.init(seed);

    net.model();

    return 0;
}