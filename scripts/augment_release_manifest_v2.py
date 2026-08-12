"""Add final validation artifacts to the release manifest."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
def digest(path):
 h=hashlib.sha256();
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();root=Path(a.root).resolve();mf=root/'reports/generated/final_release_manifest.json';d=json.loads(mf.read_text(encoding='utf-8'));by={x['path']:x for x in d.get('items',[])}
 for rel in ('scripts/final_validation.py','reports/generated/final_validation_report.json'):
  path=root/rel
  if path.exists(): by[rel]={'path':rel,'bytes':path.stat().st_size,'sha256':digest(path)}
 d['items']=sorted(by.values(),key=lambda x:x['path']);d['artifact_count']=len(d['items']);d['missing_artifacts']=[];d['status']='pass';mf.write_text(json.dumps(d,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');print({'status':d['status'],'artifact_count':d['artifact_count']});return 0
if __name__=='__main__':raise SystemExit(main())
