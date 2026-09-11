"""Verify extracted official public data against ZIP central-directory CRCs.

Reads split-archive central directories only (including ZIP64); does not extract,
delete, change E: files, or touch client data. Records a full machine-readable audit.
"""
from pathlib import Path, PurePosixPath
from concurrent.futures import ThreadPoolExecutor
import struct
import zlib
import hashlib
import json
import csv
import time
import sys
import importlib.util
from datetime import datetime, timezone


def entries(path):
    with path.open('rb') as f:
        f.seek(0,2); length=f.tell(); f.seek(max(0,length-65557)); tail=f.read()
        i=tail.rfind(b'PK\x05\x06')
        if i<0: raise ValueError('Missing end-of-central-directory')
        eocd=struct.unpack_from('<4s4H2LH',tail,i)
        _,disk,cd_disk,_,count,size,offset,_=eocd
        if count==65535 or size==0xffffffff or offset==0xffffffff:
            locator=struct.unpack_from('<4sLQL',tail,i-20)
            if locator[0]!=b'PK\x06\x07' or locator[1]!=disk:
                raise ValueError('ZIP64 index is not in final archive volume')
            f.seek(locator[2]); header=f.read(56)
            record=struct.unpack('<4sQ2H2L4Q',header)
            if record[0]!=b'PK\x06\x06': raise ValueError('Missing ZIP64 end record')
            disk,cd_disk,count,size,offset=record[4],record[5],record[7],record[8],record[9]
        if disk!=cd_disk or offset+size>length:
            raise ValueError('Central directory crosses volumes; need a dedicated archive utility')
        f.seek(offset)
        consumed=0
        for _ in range(count):
            header=f.read(46)
            row=struct.unpack('<4s6H3L5H2L',header)
            if row[0]!=b'PK\x01\x02': raise ValueError('Invalid central record')
            name_len,extra_len,comment_len=row[10:13]
            name=f.read(name_len).decode('utf-8' if row[3]&0x800 else 'cp437')
            extra=f.read(extra_len); f.read(comment_len)
            consumed+=46+name_len+extra_len+comment_len
            length=row[9]
            if length==0xffffffff:
                j=0
                while j+4<=len(extra):
                    tag,n=struct.unpack_from('<HH',extra,j)
                    if tag==1:
                        length=struct.unpack_from('<Q',extra,j+4)[0]; break
                    j+=4+n
                else: raise ValueError('Missing ZIP64 size')
            parts=PurePosixPath(name.replace('\\','/'))
            if parts.is_absolute() or '..' in parts.parts or ':' in name:
                raise ValueError('Unsafe archive path')
            if not name.endswith('/'):
                yield {'name':name,'expected_bytes':length,'expected_crc32':row[7]}
        if consumed!=size: raise ValueError('Central directory length mismatch')

def check(item):
    base,entry=item
    path=base/entry['name']
    result={**entry,'status':'ok','sha256':'','actual_bytes':0,'actual_crc32':0}
    try:
        crc=0; h=hashlib.sha256(); n=0
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):
                n+=len(chunk); crc=zlib.crc32(chunk,crc); h.update(chunk)
        result.update(actual_bytes=n,actual_crc32=crc,sha256=h.hexdigest())
        if n!=entry['expected_bytes'] or crc!=entry['expected_crc32']: result['status']='mismatch'
    except OSError as exc:
        result['status']='missing_or_unreadable'; result['error']=str(exc)
    return result

def main():
    import argparse
    from vehicle_id.attributes.modelling import metadata
    p=argparse.ArgumentParser()
    p.add_argument('--archives-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'run_metadata.json').write_text(json.dumps(metadata({'operation':'archive CRC audit','seed':36127}),indent=2))
    summaries={}
    for kind,final,base in [('surveillance','sv_data.zip',args.archives_root/'sv_data'),('web','data.zip',args.archives_root/'data')]:
        records=list(entries(args.archives_root/final)); bad=[]; seen=0
        fields=['name','expected_bytes','expected_crc32','status','actual_bytes','actual_crc32','sha256','error']
        with (args.output/f'{kind}_file_integrity.csv').open('w',newline='',encoding='utf-8') as f, ThreadPoolExecutor(max_workers=4) as pool:
            writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
            for result in pool.map(check,((base,r) for r in records)):
                writer.writerow(result); seen+=1
                if result['status']!='ok': bad.append(result)
                if seen%20000==0: print(f'{kind}: {seen}/{len(records)}, errors={len(bad)}',flush=True)
        summaries[kind]={'checked_files':seen,'errors':bad,'crc_matches_archive':not bad,
                         'notice':'Compared to supplied archive index, not publisher signature'}
        (args.output/'integrity_summary.json').write_text(json.dumps(summaries,indent=2))
    if any(v['errors'] for v in summaries.values()): raise SystemExit(2)
    print('Archive CRC audit completed.',flush=True)

if __name__=='__main__': main()
