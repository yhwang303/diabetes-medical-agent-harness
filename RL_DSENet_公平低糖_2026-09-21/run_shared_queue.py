import argparse, subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent
ap=argparse.ArgumentParser();ap.add_argument('methods',nargs='+');args=ap.parse_args()
for key in args.methods:
 subprocess.run([str(P/'.venv-native/bin/python'),str(R/'train_shared.py'),'--method',key],check=True)
print('SHARED_QUEUE_COMPLETE',flush=True)
