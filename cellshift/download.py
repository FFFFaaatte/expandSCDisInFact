"""Verified, resumable downloads; archive members are never blindly extracted."""
import argparse
import hashlib
import os
from pathlib import Path
import tarfile
import urllib.request

FILES = {
    'gbm': ('GBM_treatment.tar.gz', '8786d99f36a011cfde06b0ccabbf3690'),
    'simulated': ('simulated.tar.gz', 'd74594539ac139b65c3b2aa729a42b86'),
}

def download(url, path, md5=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (md5 is None or checksum(path) == md5):
        return path
    partial = path.with_suffix(path.suffix + '.part')
    start = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={'Range': 'bytes=%d-' % start} if start else {})
    with urllib.request.urlopen(request, timeout=120) as response:
        resumed = start > 0 and response.status == 206
        with partial.open('ab' if resumed else 'wb') as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    if md5 and checksum(partial) != md5:
        raise ValueError('Checksum mismatch: %s; remove the .part file before retrying' % partial)
    os.replace(partial, path)
    return path

def checksum(path):
    digest = hashlib.md5()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def extract(archive, destination):
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar:
            target = (root / member.name).resolve()
            if target == root and member.isdir():
                continue
            if root not in target.parents or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe archive member: %s' % member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open('wb') as out:
                    import shutil
                    shutil.copyfileobj(source, out)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['gbm', 'simulated', 'all', 'demo'])
    parser.add_argument('--root', default='data')
    args = parser.parse_args()
    root = Path(args.root)
    if args.dataset == 'demo':
        from .upstream import REVISION
        for name in ['counts.npz', 'meta_cells.csv']:
            download('https://raw.githubusercontent.com/ZhangLabGT/scDisInFact/%s/data/demo_data/%s' % (REVISION, name), root / 'demo' / name)
        return
    for key in FILES if args.dataset == 'all' else [args.dataset]:
        filename, md5 = FILES[key]
        print('Downloading and verifying', filename, flush=True)
        archive = download('https://zenodo.org/records/10472715/files/' + filename + '?download=1', root / 'archives' / filename, md5)
        marker = root / ('.' + key + '-extracted')
        if not marker.exists():
            extract(archive, root)
            marker.write_text(md5 + '\n')
        print('Ready:', key, flush=True)

if __name__ == '__main__':
    main()
