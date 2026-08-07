"""Local-filesystem stand-in for google.cloud.storage (GCS is gone — GCP billing
closed 2026-08-05). Implements exactly the surface the frozen measurement scripts
use, mapping gs://<bucket>/<path> -> $LOCAL_GCS_ROOT/<bucket>/<path>.

Put this package's parent dir FIRST on PYTHONPATH (and do NOT install the real
google-cloud-storage) — the frozen scripts then run byte-identical with all
artifact IO redirected to a local dir / Modal Volume.

Supported surface (verified against every frozen script 2026-08-06):
  storage.Client(project=...) -> Client
  Client.bucket(name) / Client.list_blobs(bucket_or_name, prefix=...)
  Bucket.blob(path) / Bucket.client
  Blob.name / .download_as_bytes() / .download_to_filename(dst)
      / .upload_from_string(data) / .upload_from_filename(src)
"""
import os
import shutil

_ROOT = os.environ.get("LOCAL_GCS_ROOT", "/vol/gcs")


class Blob:
    def __init__(self, bucket, name):
        self.bucket = bucket
        self.name = name

    @property
    def _path(self):
        return os.path.join(_ROOT, self.bucket.name, self.name)

    def exists(self, client=None):
        return os.path.exists(self._path)

    def download_as_bytes(self):
        with open(self._path, "rb") as f:
            return f.read()

    def download_to_filename(self, dst):
        shutil.copyfile(self._path, dst)

    def upload_from_string(self, data):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        if isinstance(data, str):
            data = data.encode()
        tmp = self._path + f".tmp{os.getpid()}"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, self._path)

    def upload_from_filename(self, src):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        tmp = self._path + f".tmp{os.getpid()}"
        shutil.copyfile(src, tmp)
        os.replace(tmp, self._path)


class Bucket:
    def __init__(self, client, name):
        self.client = client
        self.name = name

    def blob(self, name):
        return Blob(self, name)


class Client:
    def __init__(self, project=None):
        self.project = project

    def bucket(self, name):
        return Bucket(self, name)

    def list_blobs(self, bucket_or_name, prefix=""):
        bucket = bucket_or_name if isinstance(bucket_or_name, Bucket) else Bucket(self, bucket_or_name)
        base = os.path.join(_ROOT, bucket.name)
        root = os.path.join(base, prefix)
        # prefix may end mid-filename (GCS semantics): walk the deepest existing dir
        walk_dir = root if prefix.endswith("/") or prefix == "" else os.path.dirname(root)
        if not os.path.isdir(walk_dir):
            return iter([])
        names = []
        for dirpath, _dirs, files in os.walk(walk_dir):
            for fn in files:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, base)
                if rel.startswith(prefix) and not fn.endswith(tuple([".tmp%d" % 0])) and ".tmp" not in fn:
                    names.append(rel)
        return iter(Blob(bucket, n) for n in sorted(names))
