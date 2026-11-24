import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class MLP(nn.Module):
    def __init__(self, input_dim=44, hidden1=128, hidden2=64, output_dim=3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden2, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class Client:
    def __init__(self, model, train_data, optimizer=None, loss_fn=None):
        self.model = model
        if isinstance(model, nn.Module):
            self.train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
            self.optimizer = optimizer
            self.loss_fn = loss_fn
        else:
            self.X_train, self.y_train = train_data
            self.model_params = {'C': 1.0, 'gamma': 'scale'}

    def train_local(self, epochs=10, hospital_name=""):
        if isinstance(self.model, nn.Module):
            self.model.train()
            losses = []
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_X, batch_y in self.train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.loss_fn(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                avg_loss = epoch_loss / len(self.train_loader)
                losses.append(avg_loss)
                print(f"  {hospital_name} (MLP) - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            return losses
        else:
            self.model.fit(self.X_train, self.y_train)
            print(f"  {hospital_name} (SVM) - Trained locally")
            return None

    def compute_logits(self, comm_data):
        if isinstance(self.model, nn.Module):
            self.model.eval()
            with torch.no_grad():
                logits = self.model(comm_data)
                # Convert to probabilities for normalization
                probs = F.softmax(logits, dim=1)
            return probs
        else:
            comm_data_np = comm_data.numpy()
            decision_values = self.model.decision_function(comm_data_np)
            # Convert decision values to pseudo-probabilities
            exp_values = np.exp(decision_values - np.max(decision_values, axis=1, keepdims=True))
            probs = exp_values / exp_values.sum(axis=1, keepdims=True)
            return torch.tensor(probs, dtype=torch.float32)

    def fine_tune(self, averaged_probs, comm_data, epochs=10, hospital_name=""):
        if isinstance(self.model, nn.Module):
            self.model.train()
            losses = []
            for epoch in range(epochs):
                self.optimizer.zero_grad()
                outputs = self.model(comm_data)
                loss = - (averaged_probs * F.log_softmax(outputs, dim=1)).sum(dim=1).mean()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                print(f"  {hospital_name} (MLP) - Fine-tune Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            return losses
        else:
            comm_data_np = comm_data.numpy()
            averaged_probs_np = averaged_probs.numpy()
            labels = np.argmax(averaged_probs_np, axis=1)
            sample_weights = np.max(averaged_probs_np, axis=1)
            # Grid search for SVM parameters
            param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.01]}
            grid_search = GridSearchCV(SVC(kernel='rbf', decision_function_shape='ovr'), param_grid, cv=3)
            grid_search.fit(comm_data_np, labels, sample_weight=sample_weights)
            self.model = grid_search.best_estimator_
            self.model_params = grid_search.best_params_
            print(f"  {hospital_name} (SVM) - Fine-tuned with best params: {self.model_params}")
            return None

    def evaluate(self, test_data, test_labels):
        if isinstance(self.model, nn.Module):
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(test_data)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == test_labels).float().mean().item()
        else:
            test_data_np = test_data.numpy()
            test_labels_np = test_labels.numpy()
            predicted = self.model.predict(test_data_np)
            accuracy = (predicted == test_labels_np).mean()
        return accuracy