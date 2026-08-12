"""Refresh hashes in the release manifest after final artifact edits."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
def digest(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();root=Path(a.root).resolve();path=root/'reports/generated/final_release_manifest.json';data=json.loads(path.read_text(encoding='utf-8'));items=[];missing=[]
 for item in data.get('items',[]):
  rel=item['path'];fp=root/rel
  if fp.exists():items.append({'path':rel,'bytes':fp.stat().st_size,'sha256':digest(fp)})
  else:missing.append(rel)
 data['items']=items;data['artifact_count']=len(items);data['missing_artifacts']=missing;data['status']='pass' if not missing else 'incomplete';path.write_text(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');print({'status':data['status'],'artifact_count':len(items),'missing':missing});return 0
if __name__=='__main__':raise SystemExit(main())
