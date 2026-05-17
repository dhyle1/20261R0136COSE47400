import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import PosterDataset
from model import PosterCNN


def evaluate():
    """Evaluate trained model on test set using F1 and top-k metrics."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test_dataset = PosterDataset(
        csv_file="datasets/test_annotations.csv",
        img_dir="posters/",
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=32)

    # model
    model = PosterCNN().to(device)

    checkpoint = torch.load("best_poster_cnn.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    threshold = checkpoint["threshold"]

    model.eval()

    print(f"Using device: {device}")
    print(f"Using threshold: {threshold:.2f}")

    all_preds = []
    all_labels = []
    top3_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            # top-3 accuracy
            top3 = torch.topk(probs, k=3, dim=1).indices

            for i in range(labels.size(0)):
                true_labels = (labels[i] == 1).nonzero(as_tuple=True)[0]

                if any(t in top3[i] for t in true_labels):
                    top3_correct += 1

                total_samples += 1

    y_pred = torch.cat(all_preds).cpu().numpy()
    y_true = torch.cat(all_labels).cpu().numpy()

    micro_f1 = f1_score(y_true, y_pred, average="micro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    top3_acc = top3_correct / total_samples

    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Top-3 Accuracy: {top3_acc:.4f}")


if __name__ == "__main__":
    evaluate()