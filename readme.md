# Growing Neural Network

## Description
* Neural Network represented as directed graph with ability to grow new neurons

## Features
* New neurons and connection can be formed while learning
* Dead neurons are removed (ReLU's zero gradient)
* No need to specify number of layers/neurons when creating the network
* Rust library for fast computations on trained model (approx. 100x faster that python)

## To-Do
* Implement Adam optimizer
* Automatically remove dead neurons