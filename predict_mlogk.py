import torch
import pandas as pd
import joblib  # <-- For loading the trained scaler
from GNN_photodegradation.featurizer import Create_Dataset
from GNN_photodegradation.models.gat_model import GNNModel

# --- Define input (compound and conditions) ---
smiles = "CNC[C@@H](C1=CC(=C(C=C1)O)O)O"  # <-- smiles representation of the compound of interest
experimental_inputs = {
    'I': 2.0, # UV intensity (mW/cm²)
    'T': 25.0,  # Temperature (°C)
    'D': 0.5,   # TiO₂ dosage (g/L)
    'C0': 1.0,  # Initial concentration (mg/L)
    'pH': 7.0   # pH
}

# --- Load trained scaler ---
scaler = joblib.load("trained_scaler.pkl")

# --- Prepare and scale input ---
numerical_features = ['I', 'T', 'D', 'C0', 'pH']
numerical_values = [experimental_inputs[f] for f in numerical_features]
input_df = pd.DataFrame([numerical_values], columns=numerical_features)
scaled_values = scaler.transform(input_df)
scaled_tensor = torch.tensor(scaled_values, dtype=torch.float)  # shape: (1, 5)

# --- Create DataFrame with SMILES (for graph conversion) ---
data = pd.DataFrame([{**experimental_inputs, 'SMILES': smiles}])
dataset = Create_Dataset(data, numerical_features, scaler=scaler, inference=True)
graph, _, _ = dataset[0]  # We ignore the experimental features from dataset
print("Experimental features:", scaled_tensor)

# --- Load trained model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNNModel(GAT_input_dim=22, experimental_input_dim=scaled_tensor.shape[1])
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# --- Run prediction ---
with torch.no_grad():
    prediction, _, _ = model(graph.to(device), scaled_tensor.to(device))
    print(f"✅ Predicted -log(k) for compound of interest: {prediction.item():.4f}")

