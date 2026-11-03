"""serve/miner.py - the miner yields: SIGSTOP it while the engine works, SIGCONT when it's idle.

Maya shares this machine with a miner (SRBMiner-MULTI, started by hand outside of Maya - a tmux
session, not ours).  A request that runs next to a mining GPU is slow for as long as the miner
runs, so the moment the engine takes work - any acquire of the service's FIFO: a request, an image
encode, a context reload, an unload - the miner is stopped, and it runs again shortly after the
FIFO emptied (a small grace, so requests that queue back to back do not stop and start it twice).

Nothing here starts or kills the miner, and no pool is logged out of: only SIGSTOP / SIGCONT go to
processes whose name matches (MAYA_MINER_NAMES, default "SRBMiner-MULTI"), and only ones this
module stopped itself - a miner the user stopped by hand stays as they left it.  The pids stopped
are written to a small state file, so a server that died with the miner stopped adopts it again on
the next start (and lets it run after the grace): the miner is never left frozen for ever.  The same
file doubles as the hand-off for helpers outside Maya (a devfee daemon that unsticks a stuck miner,
for instance): while a pid is listed there, its own keeper must not SIGCONT it back into the GPU.

The miner is looked for every time, from /proc - its pid is never remembered between runs, so a
miner started, restarted or replaced while the server runs is found as it is.  And because a miner
can be started AFTER a request began (the FIFO is held by then, so no transition would look for it
a second time), a small watcher polls /proc while the engine is busy and stops one that appears - a
slower request is not left sharing the GPU with a miner that came up during it.

`attach(fifo)` wraps the FIFO of a Service (one extra line in server.py); everything it does is
caught and logged, so the miner can never take a request down with it.  One Maya server per host is
assumed (two would stop and start the miner over each other's head).

MAYA_MINER_SUSPEND=0 turns it off; MAYA_MINER_GRACE_S is the idle grace (default 5);
MAYA_MINER_POLL_S is how often /proc is looked at while the engine is busy (default 1).
"""
from __future__ import annotations

import atexit
import json
import os
import signal
import tempfile
import threading
import time

_DEFAULT_NAMES = ("srbminer-multi",)      # /proc/<pid>/comm, matched lower-case
_DEFAULT_GRACE_S = 5.0
_DEFAULT_POLL_S = 1.0


def _names() -> tuple[str, ...]:
    raw = os.environ.get("MAYA_MINER_NAMES", "").strip()
    if not raw:
        return _DEFAULT_NAMES
    return tuple(n.strip().lower() for n in raw.split(",") if n.strip()) or _DEFAULT_NAMES


def _enabled() -> bool:
    return os.environ.get("MAYA_MINER_SUSPEND", "1").strip().lower() not in ("0", "false", "no")


def _grace_s() -> float:
    raw = os.environ.get("MAYA_MINER_GRACE_S", "").strip()
    if not raw:
        return _DEFAULT_GRACE_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _DEFAULT_GRACE_S


def _poll_s() -> float:
    raw = os.environ.get("MAYA_MINER_POLL_S", "").strip()
    if not raw:
        return _DEFAULT_POLL_S
    try:
        return max(0.05, float(raw))
    except ValueError:
        return _DEFAULT_POLL_S


class MinerGuard:
    """Stops the miner while `fifo` is held (it watches, it does not own the FIFO)."""

    def __init__(self, fifo, names: tuple[str, ...] | None = None, grace_s: float | None = None,
                 state_path: str | None = None, poll_s: float | None = None):
        self.fifo = fifo
        self.names = names or _names()
        self.grace_s = _grace_s() if grace_s is None else grace_s
        self.poll_s = _poll_s() if poll_s is None else poll_s
        self.state_path = state_path or os.environ.get("MAYA_MINER_STATE") \
            or os.path.join(tempfile.gettempdir(), "maya-miner-suspend.json")
        self._lock = threading.RLock()
        self._stopped: set[int] = set()     # pids we SIGSTOPped (never a STOP that was there before us)
        self._stopped_at: float | None = None   # when the first of them was stopped (the log's duration)
        self._timer: threading.Timer | None = None
        self._watcher: threading.Thread | None = None
        self._wanted = False                # the engine is busy: the watcher has a reason to live
        self._closed = False                # the server is ending: the watcher must not stop it again
        self._adopt()
        atexit.register(self._bye)          # a clean end must not leave the miner stopped

    # ---- finding and reading the miner --------------------------------------------

    def _pids(self) -> list[int]:
        out = []
        me = os.getpid()
        for name in os.listdir("/proc"):    # (Linux only: attach checked before)
            if not name.isdigit() or int(name) == me:
                continue
            try:
                with open(f"/proc/{name}/comm", "rb") as f:
                    comm = f.read().strip().lower().decode("ascii", "ignore")
            except OSError:
                continue
            if comm in self.names:
                out.append(int(name))
        return out

    def _state(self, pid: int) -> str | None:
        """A /proc/<pid>/stat state letter (R, S, T, Z, ...): None when the process is gone."""
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                data = f.read()
            after = data.rindex(b")") + 2   # the comm can hold any letter or paren: cut after it
            return data[after:].split(None, 1)[0].decode()
        except (OSError, ValueError, IndexError):
            return None

    # ---- bookkeeping ----------------------------------------------------------------

    def _remember(self):
        try:
            with self._lock:
                pids = sorted(self._stopped)
            tmp = f"{self.state_path}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump({"stopped": pids, "at": time.time()}, f)
            os.replace(tmp, self.state_path)
        except OSError:
            pass

    def _forget(self):
        try:
            os.unlink(self.state_path)
        except OSError:
            pass

    def _adopt(self):
        """A previous server died with the miner stopped: take its pids over (and let them run again
        after the grace, unless a request arrives first) - the miner is never frozen for ever."""
        try:
            with open(self.state_path) as f:
                pids = [int(p) for p in json.load(f).get("stopped", [])]
        except (OSError, ValueError, TypeError):
            return
        ours = [p for p in pids if self._state(p) == "T"]
        if not ours:
            self._forget()
            return
        with self._lock:
            self._stopped.update(ours)
        print(f"[maya] miner: taking over the miner a previous server left stopped "
              f"({', '.join(map(str, ours))}); it runs again in {self.grace_s:.0f} s unless a request comes",
              flush=True)
        self._remember()
        self.idle()     # start the grace: it is running again soon if the engine stays idle

    # ---- the state machine ------------------------------------------------------------

    def busy(self):
        """The engine took work (the FIFO went from empty to held): the miner stops, and stays out
        while the engine works - a miner started after this point is stopped by the watcher."""
        with self._lock:
            self._cancel()
            self._wanted = True             # (set under the lock: the watcher's exit reads it there too)
            self.suspend("a request needs the engine")
            self._watch_on()

    def _watch_on(self):
        """Look at /proc now and then while the engine is busy: a miner that appears while a long
        request runs (started by hand, or restarted with a new pid) is stopped too."""
        if self._closed or (self._watcher is not None and self._watcher.is_alive()):
            return
        t = threading.Thread(target=self._watch, name="maya-miner-watch", daemon=True)
        self._watcher = t
        t.start()

    def _watch(self):
        while True:
            time.sleep(self.poll_s)
            with self._lock:                # check and stop under one lock: a suspend() that slipped
                if self._closed or not self._wanted:   # past a resume() would stop the miner for good
                    # Gone for good, and its slot cleared under the same lock _watch_on reads: a busy()
                    # that runs after this starts a new one (one that ran before it has kept this one
                    # looking - so a request never begins unwatched).
                    self._watcher = None
                    return
                try:
                    self.suspend("it was started while the engine works")   # (RLock: suspend takes it again)
                except Exception as e:      # noqa: BLE001 - the watcher must outlive a bad scan
                    print(f"[maya] miner: {type(e).__name__}: {e}", flush=True)

    def idle(self):
        """The engine has nothing left (the FIFO emptied): the miner runs again after the grace."""
        with self._lock:
            self._wanted = False
            if not self._stopped:
                return
            self._cancel()
            self._timer = threading.Timer(self.grace_s, self._grace_over)
            self._timer.daemon = True
            self._timer.start()

    def _grace_over(self):
        with self._lock:
            if self._wanted or self.fifo.locked():   # work came back during the grace: it stays stopped
                return
            self.resume()

    def _cancel(self):
        t, self._timer = self._timer, None
        if t is not None:
            t.cancel()

    def suspend(self, why: str = "the engine is busy"):
        """SIGSTOP every miner process that runs now (one we already stopped, or one the user stopped
        by hand, is left as it is)."""
        with self._lock:
            newly = []
            for pid in self._pids():
                if pid in self._stopped or self._state(pid) in (None, "T", "Z"):
                    continue                # ours already / gone / stopped by someone else / dead
                try:
                    os.kill(pid, signal.SIGSTOP)
                except OSError:
                    continue
                self._stopped.add(pid)
                newly.append(pid)
            if newly:
                if self._stopped_at is None:
                    self._stopped_at = time.monotonic()
                print(f"[maya] miner: stopped ({', '.join(map(str, newly))}) - {why}", flush=True)
                self._remember()

    def _bye(self):
        """The server is ending: no scan may stop the miner again after this."""
        self._closed = True
        self._wanted = False
        self.resume()

    def resume(self):
        """SIGCONT what we stopped - only that: a miner stopped by the user stays stopped."""
        with self._lock:
            if not self._stopped:
                return
            gone = []
            for pid in sorted(self._stopped):
                if self._state(pid) in (None, "Z"):
                    gone.append(pid)        # it ended while it was stopped
                    continue
                try:
                    os.kill(pid, signal.SIGCONT)
                except OSError as e:
                    print(f"[maya] miner: could not run pid {pid} again ({e})", flush=True)
                    continue
                print(f"[maya] miner: runs again ({pid}, stopped {time.monotonic() - self._stopped_at:.0f} s)"
                      if self._stopped_at else f"[maya] miner: runs again ({pid})", flush=True)
            self._stopped = set()
            self._stopped_at = None
            self._forget()
            if gone:
                print(f"[maya] miner: {len(gone)} stopped miner process(es) ended while stopped", flush=True)


def attach(fifo, **kw) -> bool:
    """Make `fifo` (the Service's FairLock) stop and run the miner: every path that works the engine
    goes through it.  True when it is on; it never raises."""
    try:
        if not _enabled() or os.name != "posix" or not hasattr(signal, "SIGSTOP"):
            return False
        guard = MinerGuard(fifo, **kw)

        def safe(what, *args):
            try:
                getattr(guard, what)(*args)
            except Exception as e:          # noqa: BLE001 - the miner must never take a request down
                print(f"[maya] miner: {type(e).__name__}: {e}", flush=True)

        def held_transition(old):           # acquire / wait_turn: a success that took places
            def run(*a, **k):
                was = fifo.locked()
                ok = old(*a, **k)
                if ok and not was:          # empty -> held: the engine's turn started
                    safe("busy")
                return ok
            return run

        def released(old):
            def run(*a, **k):
                was = fifo.locked()
                old(*a, **k)
                if was and not fifo.locked():   # held -> empty: the engine's turn ended
                    safe("idle")
            return run

        old_acquire, old_wait, old_release = fifo.acquire, fifo.wait_turn, fifo.release
        fifo.acquire = held_transition(old_acquire)
        fifo.wait_turn = held_transition(old_wait)
        fifo.release = released(old_release)
        return True
    except Exception as e:                  # noqa: BLE001
        print(f"[maya] miner: off ({type(e).__name__}: {e})", flush=True)
        return False
