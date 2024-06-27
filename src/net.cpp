#include <cstdlib>
#include <vector>
#include <random>
#include <iostream>

#include "../lib/net.hpp"

double relu(double x) { return std::max(0.00, x); }
double drelu(double x) { return x < 0.00 ? 0.00 : 1.00; }

double Node::bias() { return b; }
double Node::sum() { return s; }
double Node::act() { return z; }
double Node::err() { return e; }
double Node::weight(unsigned int index) { return w[index]; }

void Node::init() { s = z = e = 0.00; }
void Node::set_bias(double val) { b = val; }
void Node::set_sum(double val) { s = val; }
void Node::set_act(double val) { z = val; }
void Node::add_err(double val) { e += val; }
void Node::set_weight(unsigned int index, double val) { w[index] = val; }

unsigned int Layer::in_features() { return in; }
unsigned int Layer::out_features() { return out; }
Node *Layer::node(unsigned int index) { return &n[index]; }

void Net::add_layer(unsigned int in, unsigned int out) { layers.push_back(Layer(in, out)); }
void Net::init(std::default_random_engine &sd) {
    seed = &sd;
    std::normal_distribution<double> gaussian(0.00, 1.00);
    for(unsigned int l = 0; l < layers.size(); l++) {
        for(unsigned int n = 0; n < layers[l].out_features(); n++) {
            for(unsigned int i = 0; i < layers[l].in_features(); i++)
                layers[l].node(n)->set_weight(i, gaussian(*seed) * sqrt(2.00 / layers[l].in_features()));
        }
    }
}

unsigned int Net::num_of_layers() { return layers.size(); }
Layer *Net::layer(unsigned int index) { return &layers[index]; }
Layer *Net::back() { return &layers.back(); }

std::vector<double> Net::forward(std::vector<double> &x) {
    std::vector<double> out;
    for(unsigned int l = 0; l < layers.size(); l++) {
        for(unsigned int n = 0; n < layers[l].out_features(); n++) {
            double dot = 0.00;
            for(unsigned int i = 0; i < layers[l].in_features(); i++) {
                double weight = layers[l].node(n)->weight(i);
                dot += (l == 0 ? x[i] : layers[l-1].node(i)->act()) * weight;
            }
            dot += layers[l].node(n)->bias();

            layers[l].node(n)->init();
            layers[l].node(n)->set_sum(dot);

            if(l == layers.size() - 1) out.push_back(layers[l].node(n)->sum());
            else layers[l].node(n)->set_act(relu(layers[l].node(n)->sum()));
        }
    }
    return out;
}

void Net::train(std::vector<double> &x, std::vector<double> &y, double alpha, double lambda) {
    std::vector<double> out = forward(x);
    for(int l = layers.size() - 1; l >= 0; l--) {
        double partial_gradient = 0.00, full_gradient = 0.00;
        for(unsigned int n = 0; n < layers[l].out_features(); n++) {
            if(l == layers.size() - 1) partial_gradient = -2.00 * (y[n] - out[n]);
            else partial_gradient = layers[l].node(n)->err() * drelu(layers[l].node(n)->sum());

            double updated_bias = layers[l].node(n)->bias() - alpha * partial_gradient;
            layers[l].node(n)->set_bias(updated_bias);

            for(unsigned int i = 0; i < layers[l].in_features(); i++) {
                if(l == 0) full_gradient = partial_gradient * x[i];
                else {
                    full_gradient = partial_gradient * layers[l-1].node(i)->act();
                    layers[l-1].node(i)->add_err(partial_gradient * layers[l].node(n)->weight(i));
                }

                full_gradient += lambda * layers[l].node(n)->weight(i);

                double updated_weight = layers[l].node(n)->weight(i) - alpha * full_gradient;
                layers[l].node(n)->set_weight(i, updated_weight);
            }
        }
    }
}

void Net::model() {
    const unsigned int LIMIT = 3;
    for(unsigned int l = 0; l < layers.size(); l++) {
        if(l == LIMIT) { std::cout << "\n...\n"; continue; }
        if(l >= LIMIT && l < layers.size() - LIMIT) continue;
        std::cout << "\nL" << l << "\n";
        for(unsigned int n = 0; n < layers[l].out_features(); n++) {
            if(n == LIMIT) { std::cout << "...\n"; continue; }
            if(n >= LIMIT && n < layers[l].out_features() - LIMIT) continue;
            std::cout << "[";
            for(unsigned int i = 0; i < layers[l].in_features(); i++) {
                if(i == LIMIT) { std::cout << "... "; continue; }
                if(i >= LIMIT && i < layers[l].in_features() - LIMIT) continue;
                std::cout << layers[l].node(n)->weight(i) << " ";
            }
            std::cout << layers[l].node(n)->bias() << "b]";
            if(n != layers[l].out_features() - 1) std::cout << "\n";
        }
        std::cout << "\n(" << layers[l].in_features() << " x " << layers[l].out_features() << ")\n";
    }
    std::cout << "\n";
}

void Net::zero() {
    for(unsigned int l = 0; l < layers.size(); l++) {
        for(unsigned int n = 0; n < layers[l].out_features(); n++) {
            for(unsigned int i = 0; i < layers[l].in_features(); i++)
                layers[l].node(n)->set_weight(i, 0.00);
            layers[l].node(n)->set_bias(0.00);
        }
    }
}

void copy(Net &src, Net &dst, double tau) {
    bool empty = !dst.num_of_layers();
    for(unsigned int l = 0; l < src.num_of_layers(); l++) {
        unsigned int in = src.layer(l)->in_features();
        unsigned int out = src.layer(l)->out_features();
        if(empty) dst.add_layer(in, out);

        for(unsigned int n = 0; n < out; n++) {
            for(unsigned int i = 0; i < in; i++) {
                double src_weight = src.layer(l)->node(n)->weight(i);
                double dst_weight = dst.layer(l)->node(n)->weight(i);
                dst.layer(l)->node(n)->set_weight(i, tau * src_weight + (1.00 - tau) * dst_weight);
            }
            double src_bias = src.layer(l)->node(n)->bias();
            double dst_bias = dst.layer(l)->node(n)->bias();
            dst.layer(l)->node(n)->set_bias(tau * src_bias + (1.00 - tau) * dst_bias);
        }
    }
}

void add(Net &src, Net &dst, double tau) {
    bool empty = !dst.num_of_layers();
    for(unsigned int l = 0; l < src.num_of_layers(); l++) {
        unsigned int in = src.layer(l)->in_features();
        unsigned int out = src.layer(l)->out_features();
        if(empty) dst.add_layer(in, out);

        for(unsigned int n = 0; n < out; n++) {
            for(unsigned int i = 0; i < in; i++) {
                double src_weight = src.layer(l)->node(n)->weight(i);
                double dst_weight = dst.layer(l)->node(n)->weight(i);
                dst.layer(l)->node(n)->set_weight(i, tau * src_weight + dst_weight);
            }
            double src_bias = src.layer(l)->node(n)->bias();
            double dst_bias = dst.layer(l)->node(n)->bias();
            dst.layer(l)->node(n)->set_bias(tau * src_bias + dst_bias);
        }
    }
}