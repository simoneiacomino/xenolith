#!/usr/bin/env python3
"""Verify locked vision artifacts and inventory a GGUF without loading its tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

ROOT = Path(__file__).resolve().parent.parent

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()

def inventory(path):
    with path.open("rb") as f:
        def read(n):
            b = f.read(n)
            if len(b) != n:
                raise ValueError("truncated GGUF")
            return b
        def unpack(fmt):
            return struct.unpack("<" + fmt, read(struct.calcsize("<" + fmt)))[0]
        def string():
            n = unpack("Q")
            if n > 1048576:
                raise ValueError("oversized GGUF string")
            return read(n).decode()
        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                element, n = unpack("I"), unpack("Q")
                if n > 1048576 or element == 9:
                    raise ValueError("invalid GGUF array")
                return [value(element) for _ in range(n)]
            fmt = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}
            return unpack(fmt[kind])
        if read(4) != b"GGUF" or unpack("I") != 3:
            raise ValueError("expected GGUF v3")
        count, nmeta = unpack("Q"), unpack("Q")
        if count > 10000 or nmeta > 10000:
            raise ValueError("GGUF counts exceed limits")
        metadata = {}
        for _ in range(nmeta):
            key = string()
            if key in metadata:
                raise ValueError("duplicate GGUF key")
            metadata[key] = value(unpack("I"))
        tensors = []
        names = set()
        for _ in range(count):
            name, nd = string(), unpack("I")
            if name in names or not 1 <= nd <= 4:
                raise ValueError("invalid tensor descriptor")
            names.add(name)
            tensors.append({"name":name,"shape":[unpack("Q") for _ in range(nd)],
                            "type":unpack("I"),"offset":unpack("Q")})
        alignment = metadata.get("general.alignment", 32)
        if alignment < 1 or alignment & (alignment - 1):
            raise ValueError("invalid alignment")
        return {"metadata":metadata,"tensors":tensors,
                "data_offset":(f.tell()+alignment-1)//alignment*alignment}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=ROOT/"tests/vision/manifest.json")
    p.add_argument("--directory", type=Path, default=ROOT/".vision-cache")
    p.add_argument("--download", action="store_true")
    p.add_argument("--model", type=Path)
    p.add_argument("--inventory", type=Path)
    p.add_argument("--references", action="store_true", help="Verify/download the pinned Transformers reference files")
    a = p.parse_args()
    lock = json.loads(a.manifest.read_text())
    a.directory.mkdir(parents=True, exist_ok=True)
    for item in lock["artifacts"]:
        is_model = item["filename"].startswith("gemma-")
        path = a.model if is_model else a.directory/item["filename"]
        if path is None:
            continue
        if not path.exists() and a.download and not is_model:
            partial = path.with_suffix(".partial")
            with urllib.request.urlopen(item["url"],timeout=60) as r, partial.open("wb") as f:
                while b := r.read(8*1024*1024):
                    f.write(b)
            if partial.stat().st_size != item["size_bytes"] or digest(partial) != item["sha256"]:
                raise ValueError("download checksum mismatch")
            partial.replace(path)
        if path.stat().st_size != item["size_bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"artifact mismatch: {path}")
        print(f"verified {item['filename']} {item['sha256']}", flush=True)
        if not is_model and a.inventory:
            a.inventory.write_text(json.dumps(inventory(path),indent=2)+"\n")
    if a.references:
        ref=a.directory/"reference";ref.mkdir(parents=True,exist_ok=True)
        revision=lock['references']['transformers_commit']
        for name,expected in lock['references']['source_sha256'].items():
            if Path(name).name!=name:raise ValueError('invalid reference name')
            path=ref/name
            if not path.exists() and a.download:
                url=f'https://raw.githubusercontent.com/huggingface/transformers/{revision}/src/transformers/models/gemma4/{name}'
                with urllib.request.urlopen(url,timeout=60) as response:
                    data=response.read(4*1024*1024+1)
                if len(data)>4*1024*1024 or hashlib.sha256(data).hexdigest()!=expected:
                    raise ValueError('reference checksum mismatch')
                path.write_bytes(data)
            if digest(path)!=expected:raise ValueError(f'reference mismatch: {path}')
            print(f'verified reference {name} {expected}',flush=True)

if __name__ == "__main__":
    main()
