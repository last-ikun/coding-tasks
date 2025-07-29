import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import argparse


# === Model Definition ===
class AirQualityNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    # === Load and preprocess data ===
    parser = argparse.ArgumentParser(
        description="Train Air Quality Classification Model"
    )
    parser.add_argument("data", type=str, help="Path to the dataset CSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    X = df.drop("Air Quality", axis=1).values
    y = df["Air Quality"].values

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    # Partition data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.2, random_state=0, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=0, stratify=y_temp
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val, X_test = scaler.transform(X_val), scaler.transform(X_test)

    # Save the scaler and label encoder for the API

    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(le, "models/label_encoder.pkl")

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = AirQualityNN(input_dim=X_train.shape[1], num_classes=num_classes)
    model_save_path = "models/air_quality_model.pt"

    # === Training Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)

    # === Training Loop ===
    n_epochs = 50
    best_val_acc = 0
    for epoch in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

        # Validation metrics
        model.eval()
        all_preds = []
        all_labels = []
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                pred_labels = preds.argmax(1)
                correct += (pred_labels == yb).sum().item()
                total += yb.size(0)
                all_preds.extend(pred_labels.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        val_acc = correct / total
        val_f1 = f1_score(all_labels, all_preds, average="weighted")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
        print(
            f"Epoch {epoch + 1}/{n_epochs}: val_acc = {val_acc:.4f}, val_f1 = {val_f1:.4f}"
        )

    # === Testing ===
    # Load best model
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    correct, total = 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            pred_labels = preds.argmax(1)
            correct += (pred_labels == yb).sum().item()
            total += yb.size(0)
            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    test_acc = correct / total
    test_f1 = f1_score(all_labels, all_preds, average="weighted")
    print(f"Test accuracy: {test_acc:.4f}, Test F1 Score: {test_f1:.4f}")
