import h5py
import json
from keras.models import model_from_json, Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, InputLayer

# Load the original H5 file and extract the config and weights
with h5py.File("model.h5", "r") as f:
    config_str = f.attrs.get("model_config")
    if isinstance(config_str, bytes):
        config_str = config_str.decode("utf-8")
    config = json.loads(config_str)
    weights = {layer: f[layer][()] for layer in f if isinstance(f[layer], h5py.Dataset)}

# Recreate the model from config
model = model_from_json(json.dumps(config))

# Save the model in a new file
model.save("resaved_model.h5")

print("Model has been resaved successfully.")