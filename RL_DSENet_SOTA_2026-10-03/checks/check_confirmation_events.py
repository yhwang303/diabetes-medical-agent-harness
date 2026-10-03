"""Real concurrent event readers/writers; fixture directories, no authorization."""
import argparse
import ast
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import time

R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
import confirmation_gate as gate

PAYLOAD='atomic-fixture-'*4096
EVENTS_PER_WRITER=6


def publish(directory,writer_id):
    # Deliberately expose partial *temporary* bytes for long enough that the real
    # unlocked events() reader observes publication in progress.
    def slow_write(path,value):
        data=json.dumps(value,allow_nan=False)
        with path.open('x') as stream:
            for offset in range(0,len(data),4096):
                stream.write(data[offset:offset+4096]);stream.flush();time.sleep(.001)
    gate.write_new=slow_write
    directory=Path(directory)
    for index in range(EVENTS_PER_WRITER):
        with gate.lock(directory):
            gate.append_event(directory,dict(event='fixture_only',writer=writer_id,index=index,payload=PAYLOAD))


def observe(directory,ready,stop,result):
    directory=Path(directory);snapshots=0;pending_snapshots=0;max_count=0
    ready.set()
    try:
        while True:
            pending=bool(list((directory/'events').glob('.*.tmp')))
            rows=gate.events(directory)
            assert all(row['event']=='fixture_only' and row['payload']==PAYLOAD and
                       row['writer'] in (0,1) and 0<=row['index']<EVENTS_PER_WRITER for row in rows)
            keys={(row['writer'],row['index']) for row in rows}
            assert len(keys)==len(rows) and len(rows)>=max_count
            snapshots+=1;pending_snapshots+=int(pending);max_count=len(rows)
            if stop.is_set():break
            time.sleep(.0005)
        result.put(dict(status='passed',snapshots=snapshots,pending_snapshots=pending_snapshots,
                        final_count=max_count,keys=sorted(keys)))
    except BaseException as error:
        result.put(dict(status='failed',error=repr(error),snapshots=snapshots,pending_snapshots=pending_snapshots))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve();output.relative_to((R/'checks').resolve())
    if output.exists():raise FileExistsError(output)
    sources=[R/'confirmation_gate.py',Path(__file__).resolve()]
    source_sha={str(p.relative_to(R)):gate.sha(p) for p in sources}
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    for path in sources:ast.parse(path.read_text(),feature_version=(3,8))
    check('python38_syntax',True)
    context=mp.get_context('spawn');processes=[];stop=context.Event()
    with tempfile.TemporaryDirectory(prefix='gate_events_fixture_',dir=R/'checks') as temporary:
        directory=Path(temporary);(directory/'events').mkdir()
        queue=context.Queue();ready=context.Event()
        reader=context.Process(target=observe,args=(str(directory),ready,stop,queue));processes.append(reader)
        try:
            reader.start();check('reader_process_ready',ready.wait(5))
            writers=[context.Process(target=publish,args=(str(directory),i)) for i in range(2)]
            processes.extend(writers)
            for writer in writers:writer.start()
            for writer in writers:writer.join(10)
            check('two_real_writers_succeeded',all(w.exitcode==0 for w in writers))
            stop.set();reader.join(5);observed=queue.get(timeout=2)
            check('concurrent_reader_only_observed_complete_JSON',reader.exitcode==0 and observed['status']=='passed')
            check('reader_observed_publication_in_progress',observed['pending_snapshots']>0)
            expected={(i,j) for i in range(2) for j in range(EVENTS_PER_WRITER)}
            check('all_events_seen_once',observed['final_count']==len(expected) and {tuple(x) for x in observed['keys']}==expected)
            names=sorted(p.name for p in (directory/'events').glob('*.json'))
            check('event_numbers_contiguous_and_unique',names==['%04d.json'%i for i in range(1,len(expected)+1)])
            check('successful_publication_removes_temporaries',not list((directory/'events').glob('.*.tmp')))
            original=gate.events(directory)
            (directory/'events'/'.abandoned.tmp').write_text('{incomplete fixture')
            check('abandoned_partial_temporary_ignored',gate.events(directory)==original)
            check('abandoned_partial_temporary_retained',(directory/'events'/'.abandoned.tmp').read_text()=='{incomplete fixture')

            failed=directory/'write_failure';(failed/'events').mkdir(parents=True)
            try:
                with gate.lock(failed):gate.append_event(failed,dict(event='fixture_only',bad=float('nan')))
            except ValueError:pass
            else:raise AssertionError('Nonfinite event accepted')
            check('write_failure_not_published',gate.events(failed)==[])
            check('write_failure_keeps_temporary',len(list((failed/'events').glob('.*.tmp')))==1)
            check('write_failure_releases_lock',not (failed/'.gate.lock').exists())

            collision=directory/'collision';(collision/'events').mkdir(parents=True)
            destination=collision/'events/0002.json'
            gate.write_new(destination,dict(event='existing_fixture'))
            prior=destination.read_bytes();inode=destination.stat().st_ino
            try:
                with gate.lock(collision):gate.append_event(collision,dict(event='must_not_replace'))
            except FileExistsError:pass
            else:raise AssertionError('Existing event overwritten')
            check('publication_is_no_replace',destination.read_bytes()==prior and destination.stat().st_ino==inode)
            check('publication_collision_keeps_temporary',len(list((collision/'events').glob('.*.tmp')))==1)
            check('collision_releases_lock',not (collision/'.gate.lock').exists())
        finally:
            stop.set()
            for process in processes:
                if process.pid is not None:
                    process.join(2)
                    if process.is_alive():process.terminate();process.join(2)
            queue.close();queue.join_thread()
    check('sources_unchanged',all(gate.sha(R/name)==digest for name,digest in source_sha.items()))
    check('no_model_or_simulator_loaded',not any(m=='torch' or m.startswith('simglucose') for m in sys.modules))
    evidence=dict(status='passed',count=len(checks),checks=checks,source_sha256=source_sha,
        real_concurrent_reader=observed,real_writer_count=2,process_start_method='spawn',
        fixture_only=True,actual_selection_frozen=False,actual_confirmation_jobs_generated=False,
        training_simulation_or_remote_run=False)
    gate.write_new(output,evidence)
    print(json.dumps(dict(status='passed',count=len(checks),gate_sha256=source_sha['confirmation_gate.py'],reader=observed)))


if __name__=='__main__':main()
