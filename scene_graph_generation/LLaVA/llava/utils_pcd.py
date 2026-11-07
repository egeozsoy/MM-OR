"""
Minimal, dependency-free PCD reader — replaces open3d for MM-OR's point-cloud loading.

Why not open3d: the pinned open3d 0.17 does not support numpy 2 and drags in
dash/pydantic<2, only to read one simple file format. (trimesh does not read .pcd.)

Validated byte-exact against `open3d.io.read_point_cloud` on the released 4D-OR
pcds_sparse files (see MODERNIZATION.md).

Supports the variants the dataset actually uses, plus plain binary/ascii:
  DATA ascii | binary | binary_compressed, fields x y z [rgb], TYPE F, SIZE 4.
"""

import re
import struct

import numpy as np


def _lzf_decompress(src: bytes, expected_len: int) -> bytes:
    """
    Decompress libLZF (PCD's `binary_compressed` payload codec).

    Implemented inline because the only packaged alternative (python-lzf) is a C
    extension that would reintroduce a build dependency, and the format is small:
    a control byte is either a literal run or a back-reference.
    """
    out = bytearray(expected_len)
    ip, op, n = 0, 0, len(src)
    while ip < n:
        ctrl = src[ip]
        ip += 1
        if ctrl < 32:
            # literal run of ctrl+1 bytes
            run = ctrl + 1
            out[op:op + run] = src[ip:ip + run]
            ip += run
            op += run
        else:
            length = ctrl >> 5
            if length == 7:  # extended length
                length += src[ip]
                ip += 1
            ref = op - ((ctrl & 0x1F) << 8) - src[ip] - 1
            ip += 1
            if ref < 0:
                raise ValueError("corrupt LZF stream: negative back-reference")
            # Overlapping copies are legal and must be byte-by-byte.
            for _ in range(length + 2):
                out[op] = out[ref]
                op += 1
                ref += 1
    if op != expected_len:
        raise ValueError(f"LZF output {op} bytes, header declared {expected_len}")
    return bytes(out)


def _parse_header(fh):
    header, line = {}, b""
    while True:
        line = fh.readline()
        if not line:
            raise ValueError("unexpected EOF in PCD header")
        text = line.decode("ascii", "replace").strip()
        if not text or text.startswith("#"):
            continue
        key, _, value = text.partition(" ")
        header[key.upper()] = value.strip()
        if key.upper() == "DATA":
            return header


def read_pcd(path):
    """
    Return (xyz, rgb) as float32 arrays of shape [N, 3].

    xyz is in the file's own units (MM-OR's released clouds are millimetres; callers
    divide by 1000 — see load_point_cloud).
    rgb is normalised to 0..1 to match open3d's `.colors`; all-zeros when the file
    carries no colour channel.
    """
    with open(path, "rb") as fh:
        header = _parse_header(fh)
        fields = header.get("FIELDS", "").split()
        sizes = [int(x) for x in header.get("SIZE", "").split()]
        types = header.get("TYPE", "").split()
        counts = [int(x) for x in header.get("COUNT", "1 " * len(fields)).split()]
        n_points = int(header.get("POINTS") or (int(header["WIDTH"]) * int(header["HEIGHT"])))
        data_kind = header["DATA"].lower()

        if any(c != 1 for c in counts):
            raise NotImplementedError(f"{path}: COUNT != 1 is not supported")
        if not all(t == "F" and s == 4 for t, s in zip(types, sizes)):
            raise NotImplementedError(f"{path}: only 4-byte float fields are supported "
                                      f"(got TYPE={types} SIZE={sizes})")

        dtype = np.dtype([(f, "<f4") for f in fields])

        if data_kind == "ascii":
            arr = np.loadtxt(fh, dtype=np.float32).reshape(n_points, len(fields))
            rec = {f: arr[:, i] for i, f in enumerate(fields)}
        elif data_kind == "binary":
            rec = np.frombuffer(fh.read(n_points * dtype.itemsize), dtype=dtype, count=n_points)
        elif data_kind == "binary_compressed":
            compressed_size, uncompressed_size = struct.unpack("<II", fh.read(8))
            blob = _lzf_decompress(fh.read(compressed_size), uncompressed_size)
            # binary_compressed is COLUMN-major (all x, then all y, ...), unlike `binary`.
            rec = {}
            offset = 0
            for f in fields:
                rec[f] = np.frombuffer(blob, dtype="<f4", count=n_points, offset=offset).copy()
                offset += 4 * n_points
        else:
            raise NotImplementedError(f"{path}: DATA {data_kind} is not supported")

        xyz = np.stack([np.asarray(rec[f], dtype=np.float32) for f in ("x", "y", "z")], axis=1)

        if "rgb" in fields:
            # PCD packs r,g,b into the low 3 bytes of a float. open3d returns 0..1 floats.
            packed = np.asarray(rec["rgb"], dtype=np.float32).view(np.uint32)
            rgb = np.stack([(packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF],
                           axis=1).astype(np.float32) / 255.0
        else:
            rgb = np.zeros_like(xyz)

    return xyz, rgb


def load_point_cloud(path, scale=1000.0):
    """
    Load a released MM-OR / 4D-OR point cloud as the [N, 6] xyzrgb float32 array the
    model expects.

    `scale=1000` reproduces the `np.asarray(pc.points) / 1000` (millimetres to metres)
    that both production paths applied inline. Getting this wrong is not subtle: PTv3
    trips `assert depth <= 16` during serialization.
    """
    xyz, rgb = read_pcd(path)
    return np.concatenate([xyz / scale, rgb], axis=-1).astype(np.float32)
