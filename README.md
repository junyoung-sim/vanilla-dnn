This is a multithreaded ensemble of deep neural networks trained via mini-batch gradient descent. Specifically, N identical copies of a "main" neural network concurrently learn from a batch with N unique samples using N threads where each thread is assigned to one copy and one sample. After each copy's parameters are updated once, all corresponding parameters are averaged to update the main neural network. The aforementioned process can be repeated with many different batches by re-copying the main neural network's parameters to the ensemble.

For testing purposes, a network with over 1 million parameters was trained to predict the mean and variance of a randomly generated path following geometric Brownian motion.

| Source File | Functionality |
| --- | --- |
| [main.cpp](./src/main.cpp) | Multithreaded mini-batch training algorithm is implemented here. |
| [gbm.cpp](./src/gbm.cpp) | Geometric Brownian Motion is implemented here. |
| [net.cpp](./src/net.cpp) | Neural network class is implemented here. |

To build and execute the program:

```
make && ./exec
```