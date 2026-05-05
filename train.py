import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import PosterDataset
from model import PosterCNN
from sklearn.metrics import f1_score


def train():
    """Train PosterCNN with validation and save best model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # image transform + dataset initialization
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # enforce size (redundant here but keeps pipeline robust)
        transforms.ToTensor()
    ])

    full_dataset = PosterDataset(
        csv_file = "datasets/train_annotations.csv",
        img_dir = "posters/",
        transform = transform
    )

    # split train / validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # model + loss
    model = PosterCNN().to(device)    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    best_val_f1 = 0.0
    best_val_loss = float("inf")

    # training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # validation
        model.eval()
        val_loss = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()

        val_f1 = f1_score(y_true, y_pred, average="micro")

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1 {val_f1:.4f}"
        )

        # save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_poster_cnn.pth")
            print("Saved best model")

    print(f"Best Model — Val Loss: {best_val_loss:.4f} | Val F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    train()