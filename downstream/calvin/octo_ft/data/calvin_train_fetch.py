#!/usr/bin/env python3
"""Fetch part of CALVIN task_D_D/training straight from the official ZIP via HTTP Range.

Octo-ft needs only ~64k frames (~19 GB) of the archive. Instead of downloading
all of it, this reads the ZIP's central directory and then fetches
individual entries with Range requests. It is resumable (files already on disk
are skipped) and retries each entry independently.

All paths live under $CALVIN_ROOT (default: current directory):
  $CALVIN_ROOT/calvin_train_centraldir.json   cached central directory
  $CALVIN_ROOT/calvin_train_selection.json    written by select_frames.py
  $CALVIN_ROOT/data/task_D_D/training/        downloaded frames

Usage (in this order):
  python calvin_train_fetch.py list              # parse the central directory
  python calvin_train_fetch.py fetch 0           # metadata only (language annotations etc.)
  python select_frames.py                        # choose the training windows
  python calvin_train_fetch.py fetch_selected    # download the chosen frames
"""
import io, json, os, struct, sys, threading, time, zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter

_tls = threading.local()

URL = "http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip"
ROOT = os.path.abspath(os.environ.get("CALVIN_ROOT", "."))
CD_CACHE = f"{ROOT}/calvin_train_centraldir.json"


def _session():
    """One pooled Session per worker thread (keep-alive matters: one TCP handshake
    per entry would make 64k small fetches very slow)."""
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = "Mozilla/5.0 (CALVIN-fetch)"
        ad = HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0)
        s.mount("http://", ad); s.mount("https://", ad)
        _tls.s = s
    return s


def read_central_directory(sess):
    head = sess.head(URL, allow_redirects=True, timeout=60)
    total = int(head.headers["Content-Length"])
    if "bytes" not in head.headers.get("Accept-Ranges", "").lower():
        raise SystemExit("server does not support Range requests")
    print(f"[cd] archive {total/1e9:.2f} GB", flush=True)

    tail_n = min(4 * 1024 * 1024, total)
    tail = sess.get(URL, headers={"Range": f"bytes={total-tail_n}-{total-1}"}, timeout=120).content
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise SystemExit("EOCD not found")
    cd_size = struct.unpack("<I", tail[i+12:i+16])[0]
    cd_off = struct.unpack("<I", tail[i+16:i+20])[0]
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:      # ZIP64
        j = tail.rfind(b"PK\x06\x06")
        if j < 0:
            raise SystemExit("ZIP64 EOCD not found")
        cd_size = struct.unpack("<Q", tail[j+40:j+48])[0]
        cd_off = struct.unpack("<Q", tail[j+48:j+56])[0]
    print(f"[cd] central dir offset={cd_off} size={cd_size}", flush=True)

    cd = sess.get(URL, headers={"Range": f"bytes={cd_off}-{cd_off+cd_size-1}"}, timeout=600).content
    entries, p = [], 0
    while p < len(cd) - 4 and cd[p:p+4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[p+10:p+12])
        csize, usize = struct.unpack("<II", cd[p+20:p+28])
        nlen, elen, clen = struct.unpack("<HHH", cd[p+28:p+34])
        lho, = struct.unpack("<I", cd[p+42:p+46])
        name = cd[p+46:p+46+nlen].decode("utf-8", "replace")
        extra = cd[p+46+nlen:p+46+nlen+elen]
        if 0xFFFFFFFF in (csize, usize, lho):               # ZIP64 extra field
            q = 0
            while q < len(extra) - 4:
                hid, hsz = struct.unpack("<HH", extra[q:q+4])
                if hid == 0x0001:
                    vals, r = [], q + 4
                    for _ in range(hsz // 8):
                        vals.append(struct.unpack("<Q", extra[r:r+8])[0]); r += 8
                    k = 0
                    if usize == 0xFFFFFFFF and k < len(vals): usize = vals[k]; k += 1
                    if csize == 0xFFFFFFFF and k < len(vals): csize = vals[k]; k += 1
                    if lho == 0xFFFFFFFF and k < len(vals): lho = vals[k]; k += 1
                    break
                q += 4 + hsz
        entries.append({"name": name, "method": method, "csize": csize,
                        "usize": usize, "lho": lho})
        p += 46 + nlen + elen + clen
    print(f"[cd] {len(entries)} entries parsed", flush=True)
    return entries


def grab(e, dest_root):
    sess = _session()
    out = os.path.join(dest_root, e["name"].split("task_D_D/", 1)[-1])
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return "skip"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    last = ""
    for attempt in range(1, 6):
        try:
            hdr = sess.get(URL, headers={"Range": f"bytes={e['lho']}-{e['lho']+29}"}, timeout=120).content
            nlen, elen = struct.unpack("<HH", hdr[26:30])
            start = e["lho"] + 30 + nlen + elen
            raw = sess.get(URL, headers={"Range": f"bytes={start}-{start+e['csize']-1}"}, timeout=600).content
            data = zlib.decompress(raw, -15) if e["method"] == 8 else raw
            tmp = out + ".part"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, out)
            return "ok"
        except Exception as ex:
            last = f"{type(ex).__name__}: {str(ex)[:80]}"
            time.sleep(min(2 ** attempt, 30))
    return f"FAIL {last}"


def _run(chosen, dest):
    ok = skip = fail = 0
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(grab, e, dest): e for e in chosen}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r == "ok": ok += 1
            elif r == "skip": skip += 1
            else:
                fail += 1
                print(f"  {futs[f]['name']}: {r}", flush=True)
            if i % 2000 == 0:
                print(f"  [{i}/{len(chosen)}] ok={ok} skip={skip} fail={fail}", flush=True)
    print(f"[fetch] DONE ok={ok} skip={skip} fail={fail}", flush=True)
    print("CALVIN_TRAIN_FETCH_DONE", flush=True)


def main():
    mode = sys.argv[1]
    sess = _session()
    if mode == "list":
        entries = read_central_directory(sess)
        tr = [e for e in entries if "/training/" in e["name"] or e["name"].startswith("training/")]
        json.dump(tr, open(CD_CACHE, "w"))
        eps = sorted(e["name"] for e in tr if "episode_" in e["name"])
        meta = [e["name"] for e in tr if "episode_" not in e["name"]]
        print(f"[list] training entries={len(tr)} episodes={len(eps)} meta={len(meta)}")
        print(f"[list] first={eps[0] if eps else None}\n[list] last={eps[-1] if eps else None}")
        print("[list] metadata:", meta[:15])
        return

    tr = json.load(open(CD_CACHE))
    if mode == "fetch_selected":
        sel = json.load(open(f"{ROOT}/calvin_train_selection.json"))
        want = {f"task_D_D/training/episode_{f:07d}.npz" for f in sel["frames"]}
        meta = [e for e in tr if "episode_" not in e["name"]]
        chosen = meta + [e for e in tr if e["name"] in want]
        dest = f"{ROOT}/data/task_D_D"
        total_gb = sum(e["csize"] for e in chosen) / 1e9
        print(f"[fetch] selected {len(chosen)} entries ({len(chosen)-len(meta)} frames), "
              f"{total_gb:.1f} GB compressed", flush=True)
        _run(chosen, dest)
        return
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 64000
    dest = f"{ROOT}/data/task_D_D"
    tr = json.load(open(CD_CACHE))
    eps = sorted([e for e in tr if "episode_" in e["name"]], key=lambda e: e["name"])
    meta = [e for e in tr if "episode_" not in e["name"]]
    # contiguous slice from the start so RLDS episodes stay whole
    chosen = meta + eps[:n_frames]
    total_gb = sum(e["csize"] for e in chosen) / 1e9
    print(f"[fetch] {len(chosen)} entries ({len(meta)} meta + {min(n_frames,len(eps))} frames), "
          f"{total_gb:.1f} GB compressed", flush=True)

    _run(chosen, dest)


if __name__ == "__main__":
    main()
