# Growing Neural Networks

This README provides an overview of a directed graph-based neural network that employs backpropagation through time (BPTT) for training, where neurons are connected without traditional layers. It utilizes CUDA for efficient computation. Below, we outline the key aspects of this network, its structure, training steps, and the incorporation of BPTT.

## Network Structure

-   **Directed Graph Representation**: In this neural network, we use a directed graph to represent the connections between neurons. There are no predefined layers; instead, any neuron can be connected to any other neuron, allowing for a highly flexible architecture.

-   **Computation Order**: Neurons have a specific computation order. When a neuron is being computed, it takes its input from other neurons. Neurons are processed in a particular sequence to ensure accurate forward and backward passes.

-   **Recurrence**: If a neuron needs input from a neuron with a higher computation order, it will take the neuron's activation value from the previous iteration. This recurrence mechanism ensures that the current activation isn't used until it's computed.

## Utilizing CUDA

To efficiently utilize the power of CUDA for parallel processing, the following steps should be taken:

-   **Efficient Graph Representation**: Develop an efficient way to represent the directed graph. This representation should include the following attributes for each neuron:

    -   Index
    -   Activation Value
    -   Computation Order
    -   Bias
    -   Input Indices (neurons it receives input from)
    -   Input Weights (weights corresponding to input connections)
    -   Activation Function Identifier

-   **Training Steps**:

    1. **Move the Graph to GPU**: Transfer the neural network graph to the GPU for parallel computation.

    2. **Order Neuron Indices into Pseudo Layers**: Group neurons into pseudo layers. A pseudo layer consists of neurons that can be computed simultaneously as they don't depend on each other's activations.

    3. **Determine Grid/Block Size**: Find an optimal grid and block size for CUDA parallel processing to maximize efficiency.

    4. **Forward/Backward Pass**: Perform combined forward and backward passes while updating weights. This step includes calculating gradients and synchronizing them across the grid, followed by applying weight updates using optimization techniques such as RMSProp or other optimization algorithms.

    5. **Repeat**: Repeat steps 4 to 8 for a set number of epochs to train the network.

    6. **Update Graph (Optional)**: Periodically update the graph by adding neurons or connections. Note that this process may be slow, so it should be done sparingly.

## Backpropagation Through Time (BPTT)

Backpropagation through time (BPTT) is a key component of training this network. It extends the backpropagation algorithm to recurrent connections in the graph. In the context of this neural network:

-   BPTT is used to handle recurrent connections by unfolding the network over time and applying backpropagation to update weights over multiple time steps.

-   The recurrence mechanism, as described earlier, ensures that when a neuron needs input from a neuron with a higher computation order, it will take the neuron's activation value from the previous iteration, allowing for BPTT to work effectively.

-   BPTT is crucial for training recurrent neural networks and other networks with feedback loops to capture dependencies over time.

This directed graph-based neural network, combined with BPTT, offers a unique and flexible approach to deep learning, enabling connections between neurons without the constraints of predefined layers. Its utilization of CUDA allows for high-performance parallel computation, making it suitable for a wide range of applications and network configurations, including those with temporal dependencies.
