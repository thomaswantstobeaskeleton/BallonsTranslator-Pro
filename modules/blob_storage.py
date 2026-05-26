"""
Content-Addressed Blob Storage (CAS) — deduplicated, immutable binary storage.

Data is stored and retrieved by the SHA-256 hash of its content.
This provides natural deduplication and integrity verification.

Use cases:
  - Deduplicate intermediate images (masks, inpainted results, crops)
  - Store project snapshots without duplicating unchanged blobs
  - Reproducible pipeline outputs

Inspired by Git's object store and IPFS block store.
"""

import hashlib
import os
import os.path as osp
import shutil
import struct
from typing import Optional, Union, BinaryIO
from pathlib import Path

import numpy as np
import cv2

from utils.logger import logger as LOGGER


HASH_ALGO = "sha256"
HASH_HEX_LEN = 64


def compute_hash(data: bytes) -> str:
    """Compute the content-addressable hash of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_from_array(arr: np.ndarray) -> str:
    """Compute hash from a numpy array (for image deduplication)."""
    return compute_hash(arr.tobytes())


class BlobStorage:
    """
    On-disk content-addressed blob store.

    Directory layout (like Git objects):
      <base_dir>/
        <hash_prefix_2>/
          <hash_rest_62>

    Each file is stored uncompressed; callers can compress before writing if desired.
    """

    def __init__(self, base_dir: Union[str, Path]) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._hash_cache: dict = {}  # hash -> size (optional in-memory index)

    # --- Core API ---

    def put(self, data: bytes) -> str:
        """Store raw bytes, return the content hash (hex)."""
        h = compute_hash(data)
        path = self._hash_to_path(h)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return h

    def put_array(self, arr: np.ndarray, ext: str = ".png") -> str:
        """
        Store a numpy array as an image blob.
        Returns the hash of the encoded image bytes (not the raw array bytes).
        """
        success, encoded = cv2.imencode(ext, arr)
        if not success:
            raise ValueError(f"cv2.imencode failed for array shape {arr.shape}")
        return self.put(encoded.tobytes())

    def get(self, h: str) -> Optional[bytes]:
        """Retrieve raw bytes by hash. Returns None if not found."""
        path = self._hash_to_path(h)
        if path.exists():
            return path.read_bytes()
        return None

    def get_array(self, h: str) -> Optional[np.ndarray]:
        """Retrieve and decode an image blob by hash."""
        data = self.get(h)
        if data is None:
            return None
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        return img

    def exists(self, h: str) -> bool:
        """Check if a blob exists in storage."""
        return self._hash_to_path(h).exists()

    def delete(self, h: str) -> bool:
        """Delete a blob. Returns True if it existed and was removed."""
        path = self._hash_to_path(h)
        if path.exists():
            path.unlink()
            # Remove empty parent directory
            try:
                path.parent.rmdir()
            except OSError:
                pass
            return True
        return False

    def path(self, h: str) -> Path:
        """Return the filesystem path for a hash (whether it exists or not)."""
        return self._hash_to_path(h)

    # --- Streaming / large file support ---

    def put_stream(self, stream: BinaryIO, chunk_size: int = 65536) -> str:
        """Stream data into a temporary file, compute hash, then move into place."""
        hasher = hashlib.sha256()
        tmp_path = self.base_dir / ".tmp" / f"blob_{os.getpid()}_{id(stream)}"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "wb") as f:
            while True:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                hasher.update(chunk)
        h = hasher.hexdigest()
        dest = self._hash_to_path(h)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp_path), str(dest))
        else:
            tmp_path.unlink(missing_ok=True)
        return h

    # --- Housekeeping ---

    def walk_hashes(self):
        """Yield all stored hashes."""
        for prefix_dir in self.base_dir.iterdir():
            if not prefix_dir.is_dir() or prefix_dir.name.startswith("."):
                continue
            for blob_file in prefix_dir.iterdir():
                if blob_file.is_file() and len(blob_file.name) == HASH_HEX_LEN - 2:
                    yield prefix_dir.name + blob_file.name

    def total_size(self) -> int:
        """Return total bytes stored."""
        total = 0
        for h in self.walk_hashes():
            p = self._hash_to_path(h)
            total += p.stat().st_size
        return total

    def gc(self, reachable_hashes: set) -> int:
        """
        Garbage-collect unreachable blobs.

        Args:
            reachable_hashes: Set of hashes that must be kept.

        Returns:
            Number of blobs removed.
        """
        removed = 0
        for h in list(self.walk_hashes()):
            if h not in reachable_hashes:
                if self.delete(h):
                    removed += 1
        LOGGER.info("BlobStorage GC: removed %d unreachable blobs", removed)
        return removed

    # --- Private ---

    def _hash_to_path(self, h: str) -> Path:
        if len(h) != HASH_HEX_LEN:
            raise ValueError(f"Invalid hash length: {len(h)} (expected {HASH_HEX_LEN})")
        return self.base_dir / h[:2] / h[2:]


class ProjectBlobIndex:
    """
    Per-project index mapping logical names to blob hashes.

    This sits on top of BlobStorage and provides:
      - Logical naming: "page_001/mask" -> hash
      - Reference tracking for GC
      - Atomic batch writes

    Index format (JSON):
      {
        "version": 1,
        "entries": {
          "page_001/mask": "sha256:abc...",
          "page_001/inpainted": "sha256:def..."
        }
      }
    """

    def __init__(self, project_dir: Union[str, Path], blob_storage: BlobStorage) -> None:
        self.project_dir = Path(project_dir)
        self.blob_storage = blob_storage
        self.index_path = self.project_dir / ".blob_index.json"
        self._index: dict = {"version": 1, "entries": {}}
        self._load()

    def _load(self) -> None:
        if self.index_path.exists():
            import json
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    self._index = json.load(f)
            except Exception as e:
                LOGGER.warning("Failed to load blob index: %s", e)
                self._index = {"version": 1, "entries": {}}

    def _save(self) -> None:
        import json
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2)

    def set(self, name: str, data: bytes) -> str:
        """Store data under a logical name. Returns hash."""
        h = self.blob_storage.put(data)
        self._index["entries"][name] = h
        self._save()
        return h

    def set_array(self, name: str, arr: np.ndarray, ext: str = ".png") -> str:
        """Store a numpy array under a logical name."""
        h = self.blob_storage.put_array(arr, ext=ext)
        self._index["entries"][name] = h
        self._save()
        return h

    def get(self, name: str) -> Optional[bytes]:
        """Retrieve raw bytes by logical name."""
        h = self._index["entries"].get(name)
        if h is None:
            return None
        return self.blob_storage.get(h)

    def get_array(self, name: str) -> Optional[np.ndarray]:
        """Retrieve a numpy array by logical name."""
        h = self._index["entries"].get(name)
        if h is None:
            return None
        return self.blob_storage.get_array(h)

    def remove(self, name: str) -> bool:
        """Remove a logical entry (does not delete the blob)."""
        if name in self._index["entries"]:
            del self._index["entries"][name]
            self._save()
            return True
        return False

    def referenced_hashes(self) -> set:
        """Return all hashes currently referenced by this project."""
        return set(self._index["entries"].values())

    def names(self):
        """Yield all logical names in the index."""
        yield from self._index["entries"].keys()


def get_global_blob_storage() -> BlobStorage:
    """Return a process-global BlobStorage instance."""
    from utils import shared as C
    base = osp.join(C.PROGRAM_PATH, "blob_storage")
    return BlobStorage(base)
