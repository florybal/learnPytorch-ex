# Salvando Modelo no pytorch

```
from pathlib import Path

# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)
```

| Pytorch method | O que faz |
| --- | --- |
| torch.save | Saves a serialized object to disk using Python’s [`pickle`](https://docs.python.org/3/library/pickle.html) utility. Models, tensors and various other Python objects like dictionaries can be saved using `torch.save`. |
| torch.load | Uses `pickle`’s unpickling features to deserialize and load pickled Python object files (like models, tensors or dictionaries) into memory. You can also set which device to load the object to (CPU, GPU etc). |
| torch.nn.Module.load_state_dict | Loads a model’s parameter dictionary (`model.state_dict()`) using a saved `state_dict()` object. |

`Para carregar o modelo criado:`

```
# Instantiate a new instance of our model (this will be instantiated with random weights)
loaded_model_0 = LinearRegressionModel()

# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
```

`Para salvar os pesos do modelo:`

```
# Caminho para o diretório atual
diretorio_atual = os.getcwd()

# Diretório pai
diretorio_pai = os.path.dirname(diretorio_atual)

# Construindo o caminho
pasta_modelos = os.path.join(diretorio_pai, 'models_weights')

# Salvar os pesos do modelo na pasta
caminho_pesos = os.path.join(pasta_modelos, 'resnet18_weigths.pth')
torch.save(model_zero.state_dict(), caminho_pesos)
```