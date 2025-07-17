# GNN for Degradation Rate Constant Prediction

This project uses various graph based neural networks (Graph Neural Networks (GNNs), including Graph Attention Networks (GAT), Graph Convolutional Networks (GCN), and a combination of both) to predict the photodegradation rate constant (-logk) of small organic molecules on the TiO₂ photocatalyst. The models were originally described in a 2025 study (V. Solout, M., Ghasemi, J.B. Predicting photodegradation rate constants of water pollutants on TiO2 using graph neural network and combined experimental-graph features. Sci Rep 15, 19156 (2025). https://doi.org/10.1038/s41598-025-04220-z). In the study, the GAT model was found to outperform others, which was therefore used to construct an algorithm that can predict photo/radio-lytic degradation rate (-logk) of any small molecule drug or organic compound in aqueous medium. 


## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt


