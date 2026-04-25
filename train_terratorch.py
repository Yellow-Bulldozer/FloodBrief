"""
FloodBrief - TerraTorch CLI training wrapper.

Uses IBM's recommended TerraTorch pipeline for fine-tuning TerraMind
on Sen1Floods11. This is the officially-supported approach.

Usage:
    # Train with the provided YAML config:
    python train_terratorch.py --config configs/terramind_flood.yaml

    # Or use TerraTorch CLI directly:
    terratorch fit -c configs/terramind_flood.yaml

    # Test a trained model:
    terratorch test -c configs/terramind_flood.yaml --ckpt_path output/checkpoints/best-mIoU.ckpt

Note:
    This requires the sen1floods11_v1.1/ dataset to be downloaded and
    extracted. Update the data paths in the YAML config if needed.
"""

import os
import sys
import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(
        description="FloodBrief - TerraTorch training wrapper"
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/terramind_flood.yaml",
        help="Path to TerraTorch YAML config",
    )
    parser.add_argument(
        "--action", type=str, default="fit",
        choices=["fit", "test", "validate", "predict"],
        help="TerraTorch action: fit, test, validate, predict",
    )
    parser.add_argument(
        "--ckpt-path", type=str, default=None,
        help="Checkpoint path for test/predict",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override data directory in config",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override max_epochs in config",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch_size in config",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        print("Make sure you have the YAML config in configs/")
        sys.exit(1)

    # Build command
    cmd = ["terratorch", args.action, "-c", args.config]

    if args.ckpt_path:
        cmd.extend(["--ckpt_path", args.ckpt_path])

    # Override config values if provided
    if args.data_dir:
        print(f"[FloodBrief] Overriding data_dir -> {args.data_dir}")
        # TerraTorch allows CLI overrides of config values
        cmd.extend([
            "--data.init_args.train_data_root.S1GRD",
            f"{args.data_dir}/data/S1GRDHand",
            "--data.init_args.train_data_root.S2L1C",
            f"{args.data_dir}/data/S2L1CHand",
            "--data.init_args.train_label_data_root",
            f"{args.data_dir}/data/LabelHand",
            "--data.init_args.val_data_root.S1GRD",
            f"{args.data_dir}/data/S1GRDHand",
            "--data.init_args.val_data_root.S2L1C",
            f"{args.data_dir}/data/S2L1CHand",
            "--data.init_args.val_label_data_root",
            f"{args.data_dir}/data/LabelHand",
            "--data.init_args.test_data_root.S1GRD",
            f"{args.data_dir}/data/S1GRDHand",
            "--data.init_args.test_data_root.S2L1C",
            f"{args.data_dir}/data/S2L1CHand",
            "--data.init_args.test_label_data_root",
            f"{args.data_dir}/data/LabelHand",
            "--data.init_args.train_split",
            f"{args.data_dir}/splits/flood_train_data.txt",
            "--data.init_args.val_split",
            f"{args.data_dir}/splits/flood_valid_data.txt",
            "--data.init_args.test_split",
            f"{args.data_dir}/splits/flood_test_data.txt",
        ])

    if args.epochs:
        cmd.extend(["--trainer.max_epochs", str(args.epochs)])

    if args.batch_size:
        cmd.extend(["--data.init_args.batch_size", str(args.batch_size)])

    print(f"\n{'='*60}")
    print(f"  FloodBrief - TerraTorch Training")
    print(f"  Action:  {args.action}")
    print(f"  Config:  {args.config}")
    if args.ckpt_path:
        print(f"  Ckpt:    {args.ckpt_path}")
    print(f"{'='*60}\n")
    print(f"Running: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print("ERROR: 'terratorch' command not found.")
        print("Install TerraTorch: pip install 'terratorch>=1.2.4'")
        print()
        print("Alternatively, use the custom training script:")
        print("  python train.py --data-dir ./data/sen1floods11_v1.1")
        sys.exit(1)


if __name__ == "__main__":
    main()
