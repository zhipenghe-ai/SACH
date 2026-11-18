"""Launch SACH with the archived MIRFlickr defaults."""

from sach.cli import main as launch, run


def main(config):
    run(config, "MIRFlickr")


if __name__ == "__main__":
    launch("MIRFlickr")
