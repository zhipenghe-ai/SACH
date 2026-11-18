"""Launch the archived exploratory COCO path."""

from sach.cli import main as launch, run


def main(config):
    run(config, "COCO")


if __name__ == "__main__":
    launch("COCO")
