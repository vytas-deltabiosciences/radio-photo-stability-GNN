import pandas as pd
import torch
import joblib
from GNN_photodegradation.featurizer import Create_Dataset
from GNN_photodegradation.models.gat_model import GNNModel

# Define the compounds and their SMILES
compounds = {
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Epinephrine": "C1=CC(=C(C=C1C(C(CN)O)O)O)O",
    "Promethazine": "CN(C)CCCN1C2=CC=CC=C2SC3=CC=CC=C31",
    "Vitamin C": "C(C(C(C(=O)CO)O)O)O",
    "Amoxicillin": "CC1(C)SCC(N1C(=O)NC2=CC=C(C=C2)O)C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Acetaminophen": "CC(=O)NC1=CC=C(C=C1)O",
    "Diphenhydramine": "CN(C)CCOC(C1=CC=CC=C1)C1=CC=CC=C1",
    "Naproxen": "CC1=CC=C(C=C1)C(O)C(=O)O"
}

# Define constant experimental inputs
experimental_inputs = {
    'I': 2.0,   # UV intensity (mW/cm²)
    'T': 25.0,  # Temperature (°C)
    'D': 1.0,   # TiO₂ dosage (g/L)
    'C0': 1000.0,  # Initial concentration (mg/L)
    'pH': 7.0   # pH
}
numerical_features = ['I', 'T', 'D', 'C0', 'pH']

# Load scaler and model
scaler = joblib.load("trained_scaler.pkl")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNNModel(GAT_input_dim=22, experimental_input_dim=len(numerical_features))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Prepare results table
results = []

# Run predictions
for name, smiles in compounds.items():
    data = pd.DataFrame([{**experimental_inputs, 'SMILES': smiles}])
    dataset = Create_Dataset(data, numerical_features, scaler=scaler, inference=True)
    graph, exp_feats, _ = dataset[0]
    exp_feats = exp_feats.unsqueeze(0)

    with torch.no_grad():
        prediction, _, _ = model(graph.to(device), exp_feats.to(device))
        results.append({
            "Compound": name,
            "SMILES": smiles,
            "-log(k)": round(prediction.item(), 4)
        })

# Sort and display results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="-log(k)", ascending=True)
print(results_df.to_string(index=False))
