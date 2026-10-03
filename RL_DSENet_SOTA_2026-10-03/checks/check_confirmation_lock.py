"""Real local process lock checks only; never freezes or authorizes any selection."""
import argparse
import ast
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time

R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
import confirmation_gate as gate


def holder(directory,entered,release,events,active,peak):
    with gate.lock(Path(directory)):
        with active.get_lock():
            active.value+=1;peak.value=max(peak.value,active.value)
        try:
            events.put(dict(event='holder_enter',time_utc_seconds=time.time(),pid=os.getpid()))
            entered.set()
            if not release.wait(10):raise RuntimeError('Fixture holder was not released')
            events.put(dict(event='holder_exit',time_utc_seconds=time.time(),pid=os.getpid()))
        finally:
            with active.get_lock():active.value-=1


def waiter(directory,started,acquired,events,active,peak):
    started.set();begin=time.monotonic()
    with gate.lock(Path(directory),timeout_seconds=3.):
        with active.get_lock():
            active.value+=1;peak.value=max(peak.value,active.value)
        try:
            events.put(dict(event='waiter_enter',time_utc_seconds=time.time(),pid=os.getpid(),wait_seconds=time.monotonic()-begin))
            acquired.set()
            events.put(dict(event='waiter_exit',time_utc_seconds=time.time(),pid=os.getpid()))
        finally:
            with active.get_lock():active.value-=1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New evidence JSON under this checks directory')
    args=parser.parse_args();output=args.output.resolve()
    output.relative_to((R/'checks').resolve())
    if output.exists():raise FileExistsError(output)
    checks=[];events_seen=[];timings={}
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def timed_out(directory,seconds):
        begin=time.monotonic()
        try:
            with gate.lock(directory,timeout_seconds=seconds):raise AssertionError('Contended lock acquired')
        except TimeoutError as error:
            check('timeout_identifies_retained_lock',str(directory/'.gate.lock') in str(error) and 'retained' in str(error))
        return time.monotonic()-begin

    sources=[R/'confirmation_gate.py',Path(__file__).resolve()]
    source_sha={str(p.relative_to(R)):gate.sha(p) for p in sources}
    ast.parse(sources[0].read_text(),feature_version=(3,8))
    ast.parse(sources[1].read_text(),feature_version=(3,8))
    check('python38_syntax',True)
    check('default_wait_is_30_seconds',inspect.signature(gate.lock).parameters['timeout_seconds'].default==30.)
    context=mp.get_context('spawn');processes=[];releases=[]
    with tempfile.TemporaryDirectory(prefix='gate_lock_fixture_',dir=R/'checks') as temporary:
        directory=Path(temporary);path=directory/'.gate.lock'
        try:
            with gate.lock(directory):
                check('uncontended_lock_records_owner',path.read_text()==str(os.getpid()))
            check('normal_exit_removes_own_lock',not path.exists())
            try:
                with gate.lock(directory):raise RuntimeError('intentional fixture body error')
            except RuntimeError:pass
            check('body_error_releases_own_lock',not path.exists())

            entered=context.Event();release=context.Event();started=context.Event();acquired=context.Event()
            queue=context.Queue();releases.append(release)
            active=context.Value('i',0);peak=context.Value('i',0,lock=False)
            first=context.Process(target=holder,args=(str(directory),entered,release,queue,active,peak));processes.append(first)
            first.start();check('holder_process_acquired',entered.wait(5))
            owner=path.read_bytes()
            second=context.Process(target=waiter,args=(str(directory),started,acquired,queue,active,peak));processes.append(second)
            second.start();check('contender_process_started',started.wait(5))
            check('contender_waits_without_overlapping',not acquired.wait(.2) and path.read_bytes()==owner)
            release.set();check('contender_acquires_after_release',acquired.wait(5))
            first.join(5);second.join(5)
            check('both_contending_processes_exit_successfully',first.exitcode==0 and second.exitcode==0)
            events_seen=[queue.get(timeout=2) for _ in range(4)]
            by_event={item['event']:item for item in events_seen}
            check('two_real_processes_observed',len({e['pid'] for e in events_seen})==2)
            # Use process-shared state, not monotonic timestamps from different processes.
            check('critical_sections_do_not_overlap',peak.value==1 and active.value==0)
            check('successful_wait_is_nonzero_and_bounded',.15<=by_event['waiter_enter']['wait_seconds']<3.)
            check('contended_completion_cleans_lock',not path.exists())
            queue.close();queue.join_thread()

            entered=context.Event();release=context.Event();queue=context.Queue();releases.append(release)
            live=context.Process(target=holder,args=(str(directory),entered,release,queue,active,peak));processes.append(live)
            live.start();check('timeout_holder_acquired',entered.wait(5))
            owner=path.read_bytes();inode=path.stat().st_ino
            elapsed=timed_out(directory,.2);timings['live_owner_timeout_seconds']=elapsed
            check('live_owner_timeout_bounded',.18<=elapsed<2.)
            check('timeout_does_not_remove_live_owner_lock',path.read_bytes()==owner and path.stat().st_ino==inode and live.is_alive())
            release.set();live.join(5)
            check('original_owner_still_releases_after_timeout',live.exitcode==0 and not path.exists())
            queue.close();queue.join_thread()

            # Deliberately abandoned fixture lock; no process is allowed to remove it automatically.
            path.write_bytes(b'fixture-abandoned-owner');inode=path.stat().st_ino
            elapsed=timed_out(directory,.12);timings['abandoned_lock_timeout_seconds']=elapsed
            check('abandoned_lock_timeout_bounded',.10<=elapsed<2.)
            check('abandoned_lock_preserved_byte_for_byte',path.read_bytes()==b'fixture-abandoned-owner' and path.stat().st_ino==inode)
            path.unlink()  # Explicit test-fixture cleanup only, not gate recovery.
            begin=time.monotonic()
            try:
                with gate.lock(directory/'missing_parent',timeout_seconds=1.):raise AssertionError('Missing directory accepted')
            except FileNotFoundError:pass
            check('non_contention_errors_are_not_retried',time.monotonic()-begin<.5)
        finally:
            for event in releases:event.set()
            for process in processes:
                if process.pid is not None:
                    process.join(2)
                    if process.is_alive():process.terminate();process.join(2)
    check('source_files_unchanged',all(gate.sha(R/name)==digest for name,digest in source_sha.items()))
    check('no_model_or_simulator_imported',not any(m=='torch' or m.startswith('simglucose') for m in sys.modules))
    result=dict(status='passed',count=len(checks),checks=checks,source_sha256=source_sha,
                process_start_method='spawn',observed_lock_events=events_seen,timings=timings,
                fixture_only=True,actual_selection_frozen=False,actual_confirmation_jobs_generated=False,
                training_simulation_or_remote_run=False,default_lock_timeout_seconds=30.)
    gate.write_new(output,result)
    print(json.dumps(dict(status='passed',count=len(checks),output=str(output),gate_sha256=source_sha['confirmation_gate.py'])))


if __name__=='__main__':main()
