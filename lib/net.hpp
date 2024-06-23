#ifndef __NET_HPP_
#define __NET_HPP_

#include <cstdlib>
#include <vector>
#include <random>

double relu(double x);
double drelu(double x);

class Node
{
private:
    double b;
    double s;
    double z;
    double e;
    std::vector<double> w;
public:
    Node() {}
    Node(unsigned int in) {
        b = s = z = e = 0.00;
        w.resize(in, 0.00);
    }
    ~Node() { std::vector<double>().swap(w); }

    double bias();
    double sum();
    double act();
    double err();
    double weight(unsigned int index);

    void init();
    void set_bias(double val);
    void set_sum(double val);
    void set_act(double val);
    void add_err(double val);
    void set_weight(unsigned int index, double val);
};

class Layer
{
private:
    std::vector<Node> n;
    unsigned int in;
    unsigned int out;
public:
    Layer () {}
    Layer(unsigned int i, unsigned int o) {
        in = i; out = o;
        n.resize(out, Node(in));
    }
    ~Layer() { std::vector<Node>().swap(n); }

    unsigned int in_features();
    unsigned int out_features();

    Node *node(unsigned int index);
};

class Net
{
private:
    std::vector<Layer> layers;
    std::default_random_engine *seed;
public:
    Net() {}
    ~Net() { std::vector<Layer>().swap(layers); }

    void add_layer(unsigned int in, unsigned int out);
    void init(std::default_random_engine &sd);

    unsigned int num_of_layers();
    Layer *layer(unsigned int index);
    Layer *back();

    std::vector<double> forward(std::vector<double> &x);

    void model();
};

void copy(Net &src, Net &dst, double tau);

#endif