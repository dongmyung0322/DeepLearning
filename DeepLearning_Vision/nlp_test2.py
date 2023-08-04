import pandas as pd
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data.dataset import random_split
from tqdm import tqdm

csv_path = './data/arXiv-DataFrame.csv'
df = pd.read_csv(csv_path)

title = df['Title']
genre = df['Primary Category']
genre_dict = {}
for i in genre:
    if i not in genre_dict.values():
        genre_dict[i] = len(genre_dict)
print(genre_dict)

X_data = title.tolist()
Y_data = [genre_dict[i] for i in genre]

print(X_data[:10])
print(Y_data[:10])

# Convert labels to numerical format using LabelEncoder
label_encoder = LabelEncoder()
Y_data = label_encoder.fit_transform(Y_data)

# Tokenize the input sequences
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_data = vectorizer.fit_transform(X_data)

# Convert X_data to PyTorch tensors
X_data = torch.tensor(X_data.toarray(), dtype=torch.float32)
Y_data = torch.tensor(Y_data, dtype=torch.long)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define the genre division model using PyTorch
class GenreDivisionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GenreDivisionModel, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.softmax(self.output(x))
        return x

# Set hyperparameters
input_dim = X_data.shape[1]
hidden_dim = 256
output_dim = len(genre_dict)

# Create an instance of the model
division_model = GenreDivisionModel(input_dim, hidden_dim, output_dim)
division_model = division_model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(division_model.parameters(), lr=0.001)

# Prepare DataLoader for training and testing data
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Training loop
num_epochs = 1000
for epoch in range (num_epochs):
    division_model.train()
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = division_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluate the model
    division_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = division_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {accuracy:.4f}")

# Final evaluation
division_model.eval()
with torch.no_grad():
    outputs = division_model(x_test)
    _, predicted = torch.max(outputs.data, 1)
    test_accuracy = (predicted == y_test).sum().item() / y_test.size(0)

print(f"Final Test Accuracy: {test_accuracy:.4f}")