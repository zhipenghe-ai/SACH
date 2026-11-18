"""Launch SACH with the archived NUS-WIDE defaults."""

from sach.cli import main as launch, run


def main(config):
    run(config, "NUSWIDE")


if __name__ == "__main__":
    launch("NUSWIDE")
