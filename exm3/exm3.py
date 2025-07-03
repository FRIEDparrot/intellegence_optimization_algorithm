import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def func(x):
    return (x[:, 0]**2 + x[:, 1]**2) ** 0.25 * (torch.sin(50 * (x[:, 0]**2 + x[:, 1]**2) ** 0.1) **2 + 1)


def plot_function(func=None, title=None, is_model=False):
    # generate data
    x1_lim = [-5, 5]
    x2_lim = [-5, 5]

    x1 = np.linspace(x1_lim[0], x1_lim[1], 100).astype(np.float32)
    x2 = np.linspace(x2_lim[0], x2_lim[1], 100).astype(np.float32)

    X1, X2 = np.meshgrid(x1, x2)
    X = np.stack([X1, X2], axis=-1)  # Changed axis to -1 to get correct shape

    # Handle both function and model cases
    if is_model:
        with torch.no_grad():
            X_tensor = torch.tensor(X.reshape(-1, 2))  # Flatten for model input
            Z = func(X_tensor).detach().numpy()
            Z = Z.reshape(100, 100)  # Reshape back to grid
    else:
        Z = func(torch.tensor(X.reshape(-1, 2))).detach().cpu().numpy()
        Z = Z.reshape(100, 100)  # Reshape back to grid

    # plot the function
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('Z')
    ax1.set_title(title)

    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X1, X2, Z)
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title('Function' if not is_model else 'Prediction')
    plt.tight_layout()
    plt.show()

class SimpleMLP(nn.Module):
    def __init__(self, hidden_size=1024, lr=0.01):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.gelu1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size )
        self.gelu2 = nn.GELU()
        self.fc3 = nn.Linear(hidden_size, hidden_size )
        self.gelu3 = nn.GELU()
        self.fc4 = nn.Linear(hidden_size, 1)

        self.net = nn.Sequential(
            self.fc1,
            self.gelu1,
            self.fc2,
            self.gelu2,
            self.fc3,
            self.gelu3,
            self.fc4
        )
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # move the model to GPU if available

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        return self.net(x)

    def train_model(self, X, y, epochs=1000, batch_size=32):
        """
        Use SGD to train the model
        """
        X = X.to(self.device)
        y = y.to(self.device).unsqueeze(1)  # add a dimension for the output

        dataloader = DataLoader(list(zip(X, y)), batch_size=batch_size, shuffle=True)
        self.train()
        for epoch in range(epochs):
            for X, y in dataloader:
                self.optim.zero_grad()
                y_pred = self.forward(X)
                loss = self.loss(y_pred, y)
                loss.backward()
                self.optim.step()

            if epoch % 100 == 0:
                print( "epoch: ", epoch, "loss: ", loss.item())
                plot_function(func=self, title=f'MLP epoch {epoch}')

        # calculate the train loss
        self.eval()
        y_pred = self.predict(X)
        loss = self.loss(y_pred, y) / len(y)
        print("Average Train loss: ", loss.item())

    def test_model(self, X, y):
        """
        Test the model on the test set
        """
        self.eval()
        X = X.to(self.device)
        y = y.to(self.device).unsqueeze(1)

        y_pred = self.predict(X)
        loss = self.loss(y_pred, y) / len(y)
        print("Average Test loss: ", loss.item())

    def predict(self, X):
        return self.forward(X)


def main():
    mlp = SimpleMLP(lr=1e-4, hidden_size=1024)

    # generate 2000 data points ()
    x = torch.tensor(np.random.uniform(-5, 5, size=(2000, 2)).astype(np.float32))
    y = func(x)
    print(x.shape, y.shape)

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    mlp.train_model(X_train, y_train, epochs=1000, batch_size=32)
    mlp.test_model(X_test, y_test)

    # plot the trained model
    plot_function(func=mlp, title='MLP prediction')
    plot_function(func=func, title='True Function')

if __name__ == '__main__':
    main()
