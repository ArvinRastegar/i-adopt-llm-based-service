"""Root entry point. It parses nothing and implements nothing; the CLI owns both."""

from iadopt_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
