"""Read files out of a big remote .zip WITHOUT downloading the whole archive (HTTP range requests).

    from remote_zip import RemoteZip
    rz = RemoteZip("https://.../images.zip")
    names = rz.namelist()
    rz.extract_many(["a.jpg", "b.jpg"], out_dir, workers=8)

Only the zip's table of contents (a few MB) and the requested files are transferred.
"""
import io
import os
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

USER_AGENT = "iot-smartcar-training/1.0"


def _retry(fn, attempts=6):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            wait = 3 * (attempt + 1)
            print(f"  network error ({exc}), retrying in {wait}s…", flush=True)
            time.sleep(wait)


def _resolve(url):
    """Follow redirects once (HF / figshare redirect to a CDN) so every range request hits the final URL."""
    def go():
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            total = int(r.headers["Content-Range"].split("/")[-1])
            return r.geturl(), total
    return _retry(go)


def download(url, path, chunk=1 << 20):
    """Plain download with retries and a progress line every ~10 %."""
    def go():
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as r, open(str(path) + ".part", "wb") as f:
            total, done, step = int(r.headers.get("Content-Length") or 0), 0, 0
            while True:
                data = r.read(chunk)
                if not data:
                    break
                f.write(data)
                done += len(data)
                if total and done * 10 // total > step:
                    step = done * 10 // total
                    print(f"  {os.path.basename(str(path))}: {done / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
        os.replace(str(path) + ".part", path)
    return _retry(go)


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file object backed by HTTP range requests (with a small read-ahead buffer).
    CDN links are often signed and short-lived (figshare: 10 s), so an expired link is re-resolved."""

    def __init__(self, url, size, block=1 << 20, source_url=None):
        self.url, self.size, self.block = url, size, block
        self.source_url = source_url
        self.pos = 0
        self._buf_start, self._buf = 0, b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def _fetch(self, start, end):
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, headers={"User-Agent": USER_AGENT,
                                                                "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                if len(data) != end - start + 1:
                    raise IOError(f"short read {len(data)} / {end - start + 1}")
                return data
            except Exception as exc:
                if attempt == 5:
                    raise
                code = getattr(exc, "code", None)
                if self.source_url and code in (400, 401, 403, 404, 410):   # signed link expired
                    self.url, _ = _resolve(self.source_url)
                else:
                    time.sleep(2 * (attempt + 1))

    def prefetch(self, start, end):
        """Load exactly [start, end] into the buffer with one request (used before reading one zip member)."""
        end = min(end, self.size - 1)
        self._buf_start, self._buf = start, self._fetch(start, end)

    def read(self, n=-1):
        if self.pos >= self.size:
            return b""
        if n is None or n < 0:
            n = self.size - self.pos
        end = min(self.size, self.pos + n)
        buf_end = self._buf_start + len(self._buf)
        if not (self._buf_start <= self.pos and end <= buf_end):
            fetch_end = min(self.size, max(end, self.pos + self.block)) - 1
            self._buf_start, self._buf = self.pos, self._fetch(self.pos, fetch_end)
        data = self._buf[self.pos - self._buf_start:end - self._buf_start]
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


class RemoteZip:
    def __init__(self, url):
        self.source_url = url
        self.url, self.size = _resolve(url)
        self._local = threading.local()
        self.zip = self._zip()

    def _zip(self):
        if not hasattr(self._local, "zip"):
            self._local.zip = zipfile.ZipFile(HttpRangeFile(self.url, self.size, source_url=self.source_url))
        return self._local.zip

    def namelist(self):
        return self.zip.namelist()

    def infolist(self):
        return self.zip.infolist()

    def read(self, name):
        zf = self._zip()
        info = self._info.get(name) if hasattr(self, "_info") else None
        if info is not None:
            # local header (30 B + name + extra) + data, plus slack for a local "extra" field longer than the central one
            start = info.header_offset
            zf.fp.prefetch(start, start + 30 + len(info.filename.encode()) + len(info.extra) + info.compress_size + 1024)
        return zf.read(name)

    def extract_many(self, names, out_dir, workers=8, flatten=True, progress=True):
        """Download the listed members into out_dir (skips files already there). Returns local paths."""
        os.makedirs(out_dir, exist_ok=True)
        self._info = {i.filename: i for i in self.zip.infolist()}
        sizes = {i.filename: i.file_size for i in self.zip.infolist()}

        def target(name):
            return os.path.join(out_dir, os.path.basename(name) if flatten else name)

        todo = [n for n in names if not os.path.exists(target(n))]
        total = sum(sizes.get(n, 0) for n in todo)
        done_bytes, t0 = 0, time.time()

        def job(name):
            data = self.read(name)
            path = target(name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".part", "wb") as f:
                f.write(data)
            os.replace(path + ".part", path)
            return len(data)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(job, n) for n in todo]
            for i, f in enumerate(as_completed(futures), 1):
                done_bytes += f.result()
                if progress and (i % 50 == 0 or i == len(todo)):
                    speed = done_bytes / max(1e-6, time.time() - t0) / 1e6
                    print(f"  {i}/{len(todo)} files · {done_bytes / 1e6:.0f}/{total / 1e6:.0f} MB · {speed:.1f} MB/s",
                          flush=True)
        return [target(n) for n in names]
