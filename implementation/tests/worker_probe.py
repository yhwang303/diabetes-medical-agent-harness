"""Adversarial child programs for process-runner tests only; never in the production registry."""

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

from medical_harness.contracts import canonical, strict_json
from medical_harness.worker_entry import parent_watch, set_limits

parent_watch(int(sys.argv[1]))
probe, output_path = sys.argv[2], Path(sys.argv[3])
envelope = strict_json(sys.stdin.buffer.read(131073))
set_limits(envelope['limits'])
output_path.write_text(json.dumps({'pid':os.getpid()}))
if probe == 'crash':
    os._exit(99)
elif probe == 'cpu':
    while True:
        pass
elif probe == 'memory':
    blocks = []
    while True:
        blocks.append(bytearray(4 * 1024 * 1024))
        time.sleep(0.01)
elif probe == 'output':
    while True:
        sys.stdout.buffer.write(b'x' * 8192)
        sys.stdout.buffer.flush()
elif probe == 'malformed':
    sys.stdout.write('not-json')
elif probe == 'descendant':
    child = subprocess.Popen([sys.executable, '-I', '-c', 'import time; time.sleep(30)'])
    output_path.write_text(json.dumps({'pid':os.getpid(), 'child':child.pid}))
    time.sleep(30)
elif probe == 'hang':
    time.sleep(30)
elif probe == 'limits':
    import errno
    with output_path.with_suffix('.large').open('wb') as stream:
        try:
            stream.write(b'x' * (envelope['limits']['file_bytes'] * 2))
            stream.flush()
            file_limited = False
        except OSError as exc:
            file_limited = exc.errno == errno.EFBIG
    handles = []
    try:
        while True:
            handles.append(os.open(os.devnull, os.O_RDONLY))
    except OSError as exc:
        fd_limited = exc.errno == errno.EMFILE
    finally:
        for fd in handles:
            os.close(fd)
    result = {'pid':os.getpid(), 'ppid':os.getppid(), 'file_limited':file_limited, 'fd_limited':fd_limited,
              'cpu':resource.getrlimit(resource.RLIMIT_CPU), 'core':resource.getrlimit(resource.RLIMIT_CORE),
              'env_keys':sorted(os.environ), 'cwd':os.getcwd()}
    sys.stdout.write(canonical({'ok':True,'data':result}))
