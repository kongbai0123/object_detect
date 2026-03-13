import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=["Car"],
    max_samples=2000,
    dataset_dir="./raw"
)