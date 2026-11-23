import argparse
from pathlib import Path

from config import BASE_DIR, EPOCHS, MODEL_DIR


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate an animal image classifier.")
    parser.add_argument("--data", type=Path, default=BASE_DIR)
    parser.add_argument("--output", type=Path, default=MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--transfer-learning", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")

    from train_model import train_model

    try:
        metrics = train_model(args.data, args.output, args.epochs, args.transfer_learning)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Test loss: {metrics['loss']:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.2%}")
    print(f"Model and class labels saved to {args.output}")


if __name__ == "__main__":
    main()
