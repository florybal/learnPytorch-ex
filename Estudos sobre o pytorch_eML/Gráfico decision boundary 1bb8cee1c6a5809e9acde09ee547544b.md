# Gráfico decision boundary

[decision surface](https://machinelearningmastery.com/plot-a-decision-surface-for-machine-learning/)

```python

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min().item() - 0.1, X[:, 0].max().item() + 0.1
    y_min, y_max = X[:, 1].min().item() - 0.1, X[:, 1].max().item() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(grid).reshape(xx.shape).cpu().numpy()
    plt.contourf(xx, yy, preds, alpha=0.7, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=y.cpu().squeeze(), cmap=plt.cm.RdYlBu, edgecolors='k')
    plt.show()

```