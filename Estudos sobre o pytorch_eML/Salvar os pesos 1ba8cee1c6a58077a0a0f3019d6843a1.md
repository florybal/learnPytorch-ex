# Salvar os pesos

```python
#Obtém o diretório atual
#diretorio_atual = os.getcwd()
print(f'Diretório atual: {diretorio_atual}')

# Obtém o diretório pai
diretorio_pai = os.path.dirname(diretorio_atual)
print(f'Diretório pai: {diretorio_pai}')

# Caminho para a pasta onde os pesos do modelo serão salvos
pasta_modelos = os.path.join(diretorio_pai, 'models_weights')
print(f'Caminho para a pasta de modelos: {pasta_modelos}')

# Verifica se o diretório existe, e se não, cria o diretório
if not os.path.exists(pasta_modelos):
    print(f'O diretório "{pasta_modelos}" não existe. Criando...')
    try:
        os.makedirs(pasta_modelos)
        print(f'Diretório "{pasta_modelos}" criado com sucesso!')
    except Exception as e:
        print(f'Erro ao criar o diretório: {e}')
else:
    print(f'O diretório "{pasta_modelos}" já existe.')

# Caminho completo para salvar o arquivo de pesos
caminho_pesos = os.path.join(pasta_modelos, 'inceptionV3.pth')
print(f'Caminho completo para o arquivo de pesos: {caminho_pesos}')

# Salva os pesos do modelo no arquivo
try:
    torch.save(model_zero.state_dict(), caminho_pesos)
    print(f'Pesos do modelo salvos com sucesso em: {caminho_pesos}')
except Exception as e:
    print(f'Erro ao salvar os pesos: {e}')
```