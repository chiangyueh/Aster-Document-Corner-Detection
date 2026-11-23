                 
import argparse

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="aster")
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands = {}
    for name in ["train", "prepare", "generate", "preview", "benchmark", "evaluate", "export", "install"]:
        commands[name] = subparsers.add_parser(name)
    for name in ["train", "prepare", "generate", "preview", "benchmark", "evaluate"]:
        commands[name].add_argument("--config", default="configs/aster.yaml")

    command = commands["train"]
    command.add_argument("--set", nargs="*", default=[])
    command.add_argument("--dry-run", action="store_true")
    command.add_argument("--no-resume", action="store_true")

    command = commands["generate"]
    command.add_argument("--set", nargs="*", default=[])
    command.add_argument("--split", default="train")
    command.add_argument("--start", type=int, default=0)
    command.add_argument("--count", type=int, required=True)
    command.add_argument("--out", required=True)
    command.add_argument("--workers", type=int, default=0)

    command = commands["preview"]
    command.add_argument("--split", default="train")
    command.add_argument("--start", type=int, default=0)
    command.add_argument("--count", type=int, default=48)
    command.add_argument("--cols", type=int, default=8)
    command.add_argument("--out", default="preview.jpg")

    commands["benchmark"].add_argument("--iterations", type=int, default=8)

    command = commands["evaluate"]
    command.add_argument("--model", default="weights/Aster.pt")
    command.add_argument("--data", required=True)
    command.add_argument("--views", type=int, default=4, choices=[1, 4])
    command.add_argument("--limit", type=int)
    command.add_argument("--vis")
    command.add_argument("--out")

    command = commands["export"]
    command.add_argument("--model", default="weights/Aster.pt")
    command.add_argument("--out", default="export/Aster.onnx")
    command.add_argument("--mnn")
    command.add_argument("--opset", type=int, default=13)
    command.add_argument("--biz-code", default="Aster")

    command = commands["install"]
    command.add_argument("--model", required=True)
    command.add_argument("--target-dir", required=True)

    args = parser.parse_args(argv)
    cfg = load_config(args.config, getattr(args, "set", [])) if hasattr(args, "config") else None
    if args.command == "train":
        from .train import train, describe
        if args.dry_run:
            print(describe(cfg))
        else:
            train(cfg, resume=not args.no_resume)
    elif args.command == "prepare":
        from .assets import prepare
        prepare(cfg)
    elif args.command == "generate":
        from .data import generate_split
        generate_split(cfg, args.split, args.start, args.count, args.out, args.workers)
        print(f"generated {args.count} images: {args.out}")
    elif args.command == "preview":
        from .data import preview
        preview(cfg, args.count, args.cols, args.out, args.split, args.start)
    elif args.command == "benchmark":
        from .train import benchmark
        benchmark(cfg, args.iterations)
    elif args.command == "evaluate":
        from .evaluate import evaluate
        evaluate(args.model, args.data, cfg["eval"], args.views, args.limit, args.vis, args.out)
    elif args.command == "export":
        from .export import export_model
        export_model(args.model, args.out, args.mnn, args.opset, args.biz_code)
    elif args.command == "install":
        from .export import install_model
        install_model(args.model, args.target_dir)
