"""Execute exactly one existing frozen confirmation ticket; no method overrides."""
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ticket',type=Path,required=True)
    args=parser.parse_args()
    from evaluate_candidates import run_confirmation
    print(run_confirmation(args.ticket))


if __name__=='__main__':
    main()
