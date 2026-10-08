"""Check the extracted mirror and compare a bounded sample with official bytes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct
import zipfile
import zlib

from compcars_archive_integrity import check


def inventory(archive):
    with zipfile.ZipFile(archive) as bundle:
        records = []
        seen = set()
        for info in bundle.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or '..' in path.parts or ':' in info.filename or '\\' in info.filename:
                raise ValueError('Unsafe archive member')
            if info.filename.casefold() in seen:
                raise ValueError('Duplicate archive member')
            seen.add(info.filename.casefold())
            if not info.is_dir():
                records.append({'name': info.filename, 'expected_bytes': info.file_size,
                                'expected_crc32': info.CRC})
        return records


def official_sample(official_first_volume, mirror, limit=100):
    records = []
    with official_first_volume.open('rb') as source, zipfile.ZipFile(mirror) as bundle:
        if source.read(4) != b'PK\x07\x08':
            raise ValueError('Expected original split ZIP marker')
        while len(records) < limit:
            header = source.read(30)
            if len(header) != 30:
                raise ValueError('Official sample ended unexpectedly')
            signature, version, flags, method, mtime, mdate, crc, compressed, size, nname, nextra = struct.unpack('<4s5H3L2H', header)
            if signature != b'PK\x03\x04' or compressed == 0xffffffff or size == 0xffffffff:
                raise ValueError('Unsupported local ZIP record in bounded parity check')
            name = source.read(nname).decode('utf-8')
            source.read(nextra)
            data = source.read(compressed)
            if flags & 8:
                descriptor = source.read(16)
                if descriptor != struct.pack('<4s3L', b'PK\x07\x08', crc, compressed, size):
                    raise ValueError('Official local-header/data-descriptor disagreement')
            if name.endswith('/'):
                continue
            if not name.startswith('data/image/') or method != 8 or not flags & 1:
                raise ValueError('Unexpected entry in official image parity sample')
            # The publisher provides this archive password publicly in instruction.txt.
            decrypt = zipfile._ZipDecrypter(b'd89551fd190e38')
            decoded = decrypt(data)
            if decoded[11] != (mtime >> 8 if flags & 8 else crc >> 24):
                raise ValueError('Official archive password check failed')
            original = zlib.decompress(decoded[12:], -15)
            if len(original) != size or zlib.crc32(original) != crc:
                raise ValueError('Official decoded bytes failed CRC/size')
            copy = bundle.read(name[len('data/'):])
            if original != copy:
                raise ValueError('Mirror image differs from official download')
            records.append({'official_member': name, 'bytes': size,
                            'sha256': hashlib.sha256(original).hexdigest(), 'identical': True})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--official-first-volume', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inventory-only', action='store_true')
    args = parser.parse_args()
    entries = inventory(args.archive)
    counts = {}
    for item in entries:
        prefix = item['name'].split('/')[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    summary = {'files': len(entries), 'expanded_bytes': sum(x['expected_bytes'] for x in entries), 'top_level_file_counts': counts}
    print(json.dumps(summary), flush=True)
    if args.inventory_only:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    parity = official_sample(args.official_first_volume, args.archive)
    (args.output / 'official_image_parity.json').write_text(json.dumps({
        'matched_images': len(parity), 'sampling': 'first 100 image entries of official data.z01',
        'scope': 'Exact decompressed byte equality for this sample only; not full publisher authentication',
        'records': parity}, indent=2), encoding='utf-8')
    errors = []
    fields = ['name', 'archive_name', 'expected_bytes', 'expected_crc32', 'status',
              'actual_bytes', 'actual_crc32', 'sha256', 'error']
    with (args.output / 'web_file_integrity.csv').open('w', newline='', encoding='utf-8') as handle, ThreadPoolExecutor(max_workers=4) as pool:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, result in enumerate(pool.map(check, ((args.root / 'data/data', item) for item in entries)), 1):
            result['archive_name'] = result['name']
            # Normalize the index key expected by the existing team's audit mapper.
            result['name'] = 'data/' + result['name']
            writer.writerow(result)
            if result['status'] != 'ok':
                errors.append(result)
            if i % 25000 == 0:
                print(f'Checked {i}/{len(entries)} files; errors={len(errors)}', flush=True)
    proof = {'web': {'checked_files': len(entries), 'errors': errors, 'crc_matches_archive': not errors,
                     'source_archive': str(args.archive), 'official_sample_matches': len(parity),
                     'notice': 'All extracted files compared to third-party mirror CRCs; only 100 images compared to official bytes.'}}
    (args.output / 'integrity_summary.json').write_text(json.dumps(proof, indent=2), encoding='utf-8')
    if errors:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
