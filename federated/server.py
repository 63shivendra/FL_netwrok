import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from client import Client, MLP

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def load_and_split_data():
    df = pd.read_csv('finaldataset.csv')
    X = df.drop(['id', 'prognosis'], axis=1).values
    y = df['prognosis'].values
    # Split: 60% train, 20% communication, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_comm, X_test, y_comm, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    # Split train: 30% Hospital 1, 30% Hospital 2
    X_hosp1, X_hosp2, y_hosp1, y_hosp2 = train_test_split(X_train, y_train, test_size=0.5, stratify=y_train, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    return (
        (scaler.transform(X_hosp1), y_hosp1),
        (scaler.transform(X_hosp2), y_hosp2),
        scaler.transform(X_comm),
        (scaler.transform(X_test), y_test)
    )

def create_clients():
    (X_hosp1, y_hosp1), (X_hosp2, y_hosp2), _, _ = load_and_split_data()
    # Hospital 1: MLP
    model_1 = MLP()
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=0.0005, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    X_hosp1_tensor = torch.tensor(X_hosp1, dtype=torch.float32)
    y_hosp1_tensor = torch.tensor(y_hosp1, dtype=torch.long)
    train_data_1 = TensorDataset(X_hosp1_tensor, y_hosp1_tensor)
    client_1 = Client(model_1, train_data_1, optimizer_1, loss_fn)
    # Hospital 2: SVM with RBF kernel
    model_2 = SVC(kernel='rbf', decision_function_shape='ovr', probability=True)
    client_2 = Client(model_2, (X_hosp2, y_hosp2))
    return client_1, client_2

def compute_agreement(probs_1, probs_2):
    preds_1 = torch.argmax(probs_1, dim=1)
    preds_2 = torch.argmax(probs_2, dim=1)
    agreement = (preds_1 == preds_2).float().mean().item()
    return agreement

def main():
    num_rounds = 5
    local_epochs = 10
    (X_hosp1, y_hosp1), (X_hosp2, y_hosp2), X_comm, (X_test, y_test) = load_and_split_data()
    client_1, client_2 = create_clients()
    X_comm_tensor = torch.tensor(X_comm, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Metrics storage
    accuracies_1 = []
    accuracies_2 = []
    losses_1_train = []
    losses_1_finetune = []
    agreements = []

    print("Starting Federated Learning...\n")
    for round_num in range(num_rounds):
        print(f"Round {round_num+1}/{num_rounds}")
        
        # Local training
        losses_1 = client_1.train_local(epochs=local_epochs, hospital_name="Hospital 1")
        client_2.train_local(epochs=local_epochs, hospital_name="Hospital 2")
        if losses_1:
            losses_1_train.extend(losses_1)

        # Compute normalized probabilities on communication dataset
        probs_1 = client_1.compute_logits(X_comm_tensor)
        probs_2 = client_2.compute_logits(X_comm_tensor)

        # Average probabilities
        averaged_probs = (probs_1 + probs_2) / 2

        # Fine-tune with averaged probabilities
        losses_1_ft = client_1.fine_tune(averaged_probs, X_comm_tensor, epochs=local_epochs, hospital_name="Hospital 1")
        client_2.fine_tune(averaged_probs, X_comm_tensor, epochs=local_epochs, hospital_name="Hospital 2")
        if losses_1_ft:
            losses_1_finetune.extend(losses_1_ft)

        # Evaluate
        accuracy_1 = client_1.evaluate(X_test_tensor, y_test_tensor)
        accuracy_2 = client_2.evaluate(X_test_tensor, y_test_tensor)
        agreement = compute_agreement(probs_1, probs_2)

        accuracies_1.append(accuracy_1)
        accuracies_2.append(accuracy_2)
        agreements.append(agreement)

        print(f"  Accuracy - Hospital 1 (MLP): {accuracy_1:.4f}, Hospital 2 (SVM): {accuracy_2:.4f}")
        print(f"  Agreement on Communication Data: {agreement:.4f}\n")

    # Final Metrics
    print("Final Metrics:")
    print(f"1. Final Accuracy Hospital 1 (MLP): {accuracies_1[-1]:.4f}")
    print(f"2. Final Accuracy Hospital 2 (SVM): {accuracies_2[-1]:.4f}")
    print(f"3. Average Training Loss Hospital 1 (MLP): {np.mean(losses_1_train):.4f}")
    print(f"4. Average Fine-Tuning Loss Hospital 1 (MLP): {np.mean(losses_1_finetune):.4f}")
    print(f"5. Final Agreement on Communication Data: {agreements[-1]:.4f}")
    print(f"6. Accuracy Improvement Hospital 1: {(accuracies_1[-1] - accuracies_1[0]):.4f}")
    print(f"7. Accuracy Improvement Hospital 2: {(accuracies_2[-1] - accuracies_2[0]):.4f}")

    # Plotting
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))

    # Accuracy vs Rounds
    plt.subplot(1, 3, 1)
    plt.plot(range(1, num_rounds+1), accuracies_1, label='Hospital 1 (MLP)', color='#1f77b4')
    plt.plot(range(1, num_rounds+1), accuracies_2, label='Hospital 2 (SVM)', color='#ff7f0e')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Rounds')
    plt.legend()

    # MLP Training/Fine-Tuning Loss vs Epochs
    plt.subplot(1, 3, 2)
    total_epochs = len(losses_1_train)
    plt.plot(range(1, total_epochs+1), losses_1_train, label='Training Loss', color='#1f77b4')
    plt.plot(range(1, total_epochs+1), losses_1_finetune, label='Fine-Tuning Loss', color='#ff7f0e')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MLP Loss Over Epochs')
    plt.legend()

    # Agreement vs Rounds
    plt.subplot(1, 3, 3)
    plt.plot(range(1, num_rounds+1), agreements, label='Communication Data Agreement', color='#1f77b4')
    plt.xlabel('Round')
    plt.ylabel('Agreement')
    plt.title('Model Agreement Over Rounds')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()