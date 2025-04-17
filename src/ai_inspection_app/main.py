import argparse
from ai_inspection_app.models.detector import train_detector, infer_detector

def main():
    parser = argparse.ArgumentParser(description="AI Inspection App CLI")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=4)

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--image-path", type=str, required=True)

    args = parser.parse_args()

    if args.command == "train":
        train_detector(epochs=args.epochs, batch_size=args.batch_size)
    elif args.command == "infer":
        infer_detector(image_path=args.image_path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()