Federated Learning for Healthcare: Hospital Model Collaboration
Project Overview
This project implements a federated learning system simulating two hospitals collaborating to improve their machine learning models without sharing raw patient data. The system uses a client-server architecture where:

Hospital 1 trains a Multi-Layer Perceptron (MLP) using PyTorch.
Hospital 2 trains a Support Vector Machine (SVM) with an RBF kernel using scikit-learn.
The central server aggregates model outputs (probabilities) from both hospitals on a shared communication dataset and distributes averaged probabilities for fine-tuning.
The process runs iteratively over 5 rounds, with 10 epochs of local training and fine-tuning per round, aiming to improve model accuracy.

The dataset (finaldataset.csv) contains 66,180 rows and 46 columns, including patient id, prognosis (target with values 0, 1, 2), and 44 features. The system ensures privacy by exchanging only model probabilities, not raw data.
Objectives

Train distinct models (MLP and SVM) on separate hospital datasets.
Use a shared communication dataset for collaborative learning via probability averaging.
Evaluate performance on a separate test dataset.
Track 7 metrics and generate 3 visualization charts to monitor accuracy and model agreement.

Repository Structure

client.py: Defines the Client class, handling model training, probability computation, fine-tuning, and evaluation for both MLP and SVM models.
server.py: Manages the federated learning process, data splitting, client coordination, probability aggregation, and metric visualization.
finaldataset.csv: Input dataset (not included; user must provide with 46 columns, including id, prognosis, and 44 features).
README.md: This file, providing project documentation and instructions.

Prerequisites
Software Requirements

Python 3.8+
Required libraries:pip install torch pandas numpy scikit-learn matplotlib



Dataset Requirements

File: finaldataset.csv
Format: CSV with 66,180 rows and 46 columns:
id: Unique patient identifier (object).
prognosis: Target variable (int64, values 0, 1, 2).
44 feature columns (float64, prefixed with obs_).


Ensure the dataset is placed in the same directory as client.py and server.py.

Setup Instructions

Clone or Set Up the Repository:

Create a directory (e.g., C:\Users\hexlive63\OneDrive\Desktop\final_fl).
Save client.py and server.py in this directory.
Place finaldataset.csv in the same directory.


Install Dependencies:
pip install torch pandas numpy scikit-learn matplotlib


Verify Dataset:

Ensure finaldataset.csv matches the expected format (46 columns, no missing values).
Check that prognosis has values 0, 1, 2 for multi-class classification.


Run the Program:
python server.py


The script will execute the federated learning process, print epoch details, metrics, and display charts.



Code Details
Data Splitting

Training Data: 60% of the dataset (39,708 rows), split equally:
Hospital 1: 30% (19,854 rows).
Hospital 2: 30% (19,854 rows).


Communication Dataset: 20% (13,236 rows), shared for probability exchange.
Test Dataset: 20% (13,236 rows), used for final evaluation.
Stratified sampling ensures balanced prognosis distribution across splits.

Models

Hospital 1 (MLP):
Architecture: 44 input features → 128 hidden units (ReLU, dropout 0.3) → 64 hidden units (ReLU, dropout 0.3) → 3 output classes.
Optimizer: Adam (learning rate 0.0005, weight decay 1e-4).
Loss: Cross-entropy for local training, KL divergence for fine-tuning.


Hospital 2 (SVM):
Model: SVM with RBF kernel, decision_function_shape='ovr', probability=True.
Fine-tuning: Grid search over C=[0.1, 1, 10] and gamma=['scale', 'auto', 0.01] with sample weights.



Federated Learning Process

Local Training: Each hospital trains its model for 10 epochs on its private data.
Probability Computation: Hospitals compute softmax probabilities on the communication dataset and send them to the server.
Aggregation: The server averages probabilities from both hospitals.
Fine-Tuning: Hospitals fine-tune their models for 10 epochs using averaged probabilities (MLP uses KL divergence; SVM retrains with pseudo-labels and weights).
Evaluation: Accuracy is computed on the test dataset after each round.
Iterations: 5 rounds of the above steps.

Metrics

Final Accuracy Hospital 1 (MLP): Accuracy on test dataset after final round.
Final Accuracy Hospital 2 (SVM): Accuracy on test dataset after final round.
Average Training Loss Hospital 1 (MLP): Mean cross-entropy loss across all local training epochs.
Average Fine-Tuning Loss Hospital 1 (MLP): Mean KL divergence loss across all fine-tuning epochs.
Final Agreement on Communication Data: Percentage of matching predictions between MLP and SVM on the communication dataset.
Accuracy Improvement Hospital 1: Final accuracy minus initial accuracy for MLP.
Accuracy Improvement Hospital 2: Final accuracy minus initial accuracy for SVM.

Charts

Accuracy Over Rounds: Line plot showing MLP and SVM accuracies across 5 rounds.
MLP Loss Over Epochs: Line plot of training and fine-tuning losses for Hospital 1.
Model Agreement Over Rounds: Line plot of prediction agreement on the communication dataset.

Expected Output

Console:
Epoch details for Hospital 1 (MLP) during training and fine-tuning (e.g., Hospital 1 (MLP) - Epoch 1/10, Loss: 0.6829).
Per-round accuracies and agreement (e.g., Accuracy - Hospital 1 (MLP): 0.7919, Hospital 2 (SVM): 0.7863).
Final metrics (e.g., Final Accuracy Hospital 1 (MLP): 0.7919).


Charts:
Three matplotlib plots displayed at the end, showing accuracy, loss, and agreement trends.



Troubleshooting Accuracy Issues
If accuracy improvements are minimal (e.g., previous run showed 0.0060 for MLP):

Normalization: The code uses softmax probabilities instead of raw logits/decision values, ensuring compatibility between MLP and SVM outputs.
Enhanced Models: MLP has a larger architecture with dropout; SVM uses grid search for optimal parameters.
Shared Communication Dataset: Reduces distribution shifts compared to separate communication datasets.
More Epochs: 10 epochs per round for training and fine-tuning allow better convergence.
Suggestions for Further Improvement:
Increase num_rounds to 10 in server.py.
Adjust MLP learning rate (e.g., lr=0.0001) in create_clients.
Expand SVM param_grid (e.g., add C=[0.01, 100]) in client.py.
Combine cross-entropy and KL divergence losses for MLP fine-tuning.



Running the Code

Navigate to the project directory:cd C:\Users\hexlive63\OneDrive\Desktop\final_fl


Run the server script:python server.py


Verify outputs:
Console logs for epoch details and metrics.
Matplotlib windows showing three charts.



Notes

The code assumes finaldataset.csv has the exact structure described (46 columns, no missing values).
If errors occur, verify library versions or dataset format.
For further customization, adjust num_rounds, local_epochs, or model hyperparameters in server.py and client.py.

License
This project is for educational purposes and not licensed for commercial use. Ensure compliance with data privacy regulations when using real healthcare data.
