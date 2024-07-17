This is a multithreaded ensemble of neural networks trained via mini-batch gradient descent. Specifically, N identical copies of a "main" neural network concurrently learn from a batch with N unique samples using N threads where each thread is assigned to one copy and one sample. After each copy's parameters are updated once, all corresponding parameters are averaged to update the main neural network. The aforementioned process can be repeated with many different batches by re-copying the main neural network's parameters to the ensemble.

As an example (for testing purposes only), a neural network with 1M+ parameters was trained 10 times given a batch with 10 samples to estimate the mean and variance of a randomly generated path that follows geometric Brownian motion.

![alt text](https://github.com/junyoung-sim/vanilla-nn/blob/main/res/fig1.png)

![alt text](https://github.com/junyoung-sim/vanilla-nn/blob/main/res/fig2.png)

![alt text](https://github.com/junyoung-sim/vanilla-nn/blob/main/res/fig3.png)