"""Train YOLO-World weights and return the selected best checkpoint."""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLOWorld


def parse_device(value):
    """Convert e.g. '0,1' to the list expected by Ultralytics."""
    value = str(value).strip()
    if value.lower() == "cpu":
        return "cpu"
    if all(part.strip().isdigit() for part in value.split(",")):
        devices = [int(part.strip()) for part in value.split(",")]
        return devices[0] if len(devices) == 1 else devices
    return value


def train_model(*, model_config, pretrained, data, epochs=75, imgsz=1536, device="0,1",
                batch=8, project="runs/detect", name="conformal_pipeline"):
    """Train and return (model, best_weights_path)."""
    model_config, pretrained, data = map(str, (model_config, pretrained, data))
    for required in (model_config, pretrained, data):
        if not Path(required).exists():
            raise FileNotFoundError(f"Required training input not found: {required}")

    torch.use_deterministic_algorithms(False)
    model = YOLOWorld(model_config)
    model.load(pretrained)
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        device=parse_device(device),
        batch=batch,
        project=project,
        name=name,
    )

    trainer = getattr(model, "trainer", None)
    if trainer is None:
        raise RuntimeError("Ultralytics did not expose the training checkpoint location")
    best_weights = Path(trainer.best)
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training completed but best checkpoint is missing: {best_weights}")
    return model, best_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True, help="YOLO-World model YAML")
    parser.add_argument("--pretrained", required=True, help="Initial .pt weights")
    parser.add_argument("--data", required=True, help="Ultralytics data YAML with train and val splits")
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--device", default="0,1", help="GPU ids such as 0,1, or cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="conformal_pipeline")
    args = parser.parse_args()

    _, best = train_model(
        model_config=args.model_config,
        pretrained=args.pretrained,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )
    print(f"Best model weights: {best}")


if __name__ == "__main__":
    main()
