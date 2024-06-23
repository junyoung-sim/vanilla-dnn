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
    std::vector<double> yhat;
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

            if(l == layers.size() - 1) yhat.push_back(layers[l].node(n)->sum());
            else layers[l].node(n)->set_act(relu(layers[l].node(n)->sum()));
        }
    }

    return yhat;
}

void Net::model() {
    for(unsigned int l = 0; l < layers.size(); l++) {
        std::cout << "\n";
        std::cout << "L" << l << ": [";
        for(unsigned int n = 0; n < layers[l].out_features(); n++) {
            if(n != 0) std::cout << "     ";
            std::cout << "[";
            for(unsigned int i = 0; i < layers[l].in_features(); i++) {
                std::cout << layers[l].node(n)->weight(i) << " ";
            }
            std::cout << layers[l].node(n)->bias() << "b]";
            if(n != layers[l].out_features() - 1) std::cout << "\n";
        }
        std::cout << "]\n";
        std::cout << "(" << layers[l].in_features() << " x " << layers[l].out_features() << ")\n";
    }
    std::cout << "\n";
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

/*
void DDPG::optimize_critic(std::vector<double> &state_action, double q, double optimal, std::vector<double> &agrad, std::vector<bool> &flag, double alpha, double lambda) {
    for(int l = critic->num_of_layers() - 1; l >= 0; l--) {
        double part = 0.00, grad = 0.00;
        for(unsigned int n = 0; n < critic->layer(l)->out_features(); n++) {
            if(l == critic->num_of_layers() - 1) part = -2.00 * (optimal - q);
            else part = critic->layer(l)->node(n)->err() * drelu(critic->layer(l)->node(n)->sum());

            double updated_bias = critic->layer(l)->node(n)->bias() - alpha * part;
            critic->layer(l)->node(n)->set_bias(updated_bias);

            for(unsigned int i = 0; i < critic->layer(l)->in_features(); i++) {
                if(l == 0) {
                    grad = part * state_action[i];
                    if(i < agrad.size()) {
                        agrad[i] = part * critic->layer(l)->node(n)->weight(i);
                        flag[i] = true;
                    }
                }
                else {
                    grad = part * critic->layer(l-1)->node(i)->act();
                    critic->layer(l-1)->node(i)->add_err(part * critic->layer(l)->node(n)->weight(i));
                }

                grad += lambda * critic->layer(l)->node(n)->weight(i);

                double updated_weight = critic->layer(l)->node(n)->weight(i) - alpha * grad;
                critic->layer(l)->node(n)->set_weight(i, updated_weight);
            }
        }
    }
}

void DDPG::optimize_actor(std::vector<double> &state, std::vector<double> &action, std::vector<double> &agrad, std::vector<bool> &flag, double alpha, double lambda) {
    for(int l = actor->num_of_layers() - 1; l >= 0; l--) {
        double part = 0.00, grad = 0.00;
        for(unsigned int n = 0; n < actor->layer(l)->out_features(); n++) {
            if(l == actor->num_of_layers() - 1) {
                while(!flag[n]) {}
                part = agrad[n] * action[n] * (1.00 - action[n]);
            }
            else part = actor->layer(l)->node(n)->err() * drelu(actor->layer(l)->node(n)->sum());

            double updated_bias = actor->layer(l)->node(n)->bias() + alpha * part;
            actor->layer(l)->node(n)->set_bias(updated_bias);

            for(unsigned int i = 0; i < actor->layer(l)->in_features(); i++) {
                if(l == 0) grad = part * state[i];
                else {
                    grad = part * actor->layer(l-1)->node(i)->act();
                    actor->layer(l-1)->node(i)->add_err(part * actor->layer(l)->node(n)->weight(i));
                }

                grad += lambda * actor->layer(l)->node(n)->weight(i);

                double updated_weight = actor->layer(l)->node(n)->weight(i) + alpha * grad;
                actor->layer(l)->node(n)->set_weight(i, updated_weight);
            }
        }
    }
}

double DDPG::optimize(Memory &memory, double gamma, double alpha, double lambda) {
    std::vector<double> *state = memory.state();
    std::vector<double> *action = memory.action();

    std::vector<double> state_action;
    state_action.insert(state_action.end(), action->begin(), action->end());
    state_action.insert(state_action.end(), state->begin(), state->end());

    std::vector<double> *next_state = memory.next_state();
    std::vector<double> next_state_action = target_actor.forward(*next_state, false);
    next_state_action.insert(next_state_action.end(), next_state->begin(), next_state->end());

    std::vector<double> q = critic->forward(state_action, false);
    std::vector<double> future = target_critic.forward(next_state_action, false);
    double optimal = memory.reward() + gamma * future[0];

    std::vector<double> agrad(action->size(), 0.00);
    std::vector<bool> flag(action->size(), false);

    std::thread critic_optimizer(&DDPG::optimize_critic, this, std::ref(state_action),
                                 q[0], optimal, std::ref(agrad), std::ref(flag), alpha, lambda);
    std::thread actor_optimizer(&DDPG::optimize_actor, this, std::ref(*state),
                                std::ref(*action), std::ref(agrad), std::ref(flag), alpha, lambda);

    critic_optimizer.join();
    actor_optimizer.join();

    return q[0];
}
*/