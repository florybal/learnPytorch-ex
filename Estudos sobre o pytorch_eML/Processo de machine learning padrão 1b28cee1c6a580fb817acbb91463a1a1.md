# Processo de machine learning padrão

1. Get data ready
2. Build or pick a pretrained model
    
    2.1 Pick a loss function & optmizer
    
    2.2 Build a training loop 
    
3. fit the model to the data and make prediction
4. evaluate the model 
5. Improve through experimentation
6. Save and reload your trained model

1. Preparando a data
    
    Separando dados em %80 para o x e %20 para o y
    
    ```python
    
    train_split = int(0.8*len(X))
    x_train, y_train= X[:train_split], y[:train_split]
    x_test, y_test = X[train_split:], y[train_split:]
    
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    len(x_test), len(x_train), len(y_test), len(y_train)
    ```
    
2. Criando o modelo
    
    ```python
    class MoonModelV0(nn.Module):
        def __init__(self, in_features, out_features, hidden_units):
            super().__init__()
            
            self.layer1 = nn.Linear(in_features=in_features, 
                                     out_features=hidden_units)
            self.layer2 = nn.Linear(in_features=hidden_units, 
                                     out_features=hidden_units)
            self.layer3 = nn.Linear(in_features=hidden_units,
                                    out_features=out_features)
            self.relu = nn.ReLU()
    
        def forward(self, x): #como a rede computa( o que vai fazer com o dado x)
    	# computation goes through layer_1 first then the output of layer_1 goes through layer_2
            return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))
    ```
    
3. Treinamento do modelo
    
    ```python
    def train(model, dataloader, lossfunc, optimizer):
      model.train()
      cumloss = 0.0
      for X, y in dataloader:
        X = X.unsqueeze(1).float().to(device)
        y = y.unsqueeze(1).float().to(device)
    
        pred = model(X)
        loss = lossfunc(pred, y)
    
        # zera os gradientes acumulados
        optimizer.zero_grad()
        # computa os gradientes
        loss.backward()
        # anda, de fato, na direção que reduz o erro local
        optimizer.step()
    
        # loss é um tensor; item pra obter o float
        cumloss += loss.item()
    
      return cumloss / len(dataloader)
    
    def test(model, dataloader, lossfunc):
      model.eval()
      cumloss = 0.0
      
      with torch.no_grad():
        for X, y in dataloader:
          X = X.unsqueeze(1).float().to(device)
          y = y.unsqueeze(1).float().to(device)
    
          pred = model(X)
          loss = lossfunc(pred, y)
          cumloss += loss.item()
    
      return cumloss / len(dataloader)
    ```