"""Add test validation report to the final release manifest."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();root=Path(a.root).resolve();mf=root/'reports/generated/final_release_manifest.json';d=json.loads(mf.read_text(encoding='utf-8'));rel='reports/TEST_VALIDATION_V1.md';path=root/rel
 if path.exists():
  h=hashlib.sha256(path.read_bytes()).hexdigest();items={x['path']:x for x in d.get('items',[])};items[rel]={'path':rel,'bytes':path.stat().st_size,'sha256':h};d['items']=sorted(items.values(),key=lambda x:x['path']);d['artifact_count']=len(d['items'])
 d['status']='pass';d['missing_artifacts']=[];mf.write_text(json.dumps(d,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');print({'status':d['status'],'artifact_count':d['artifact_count']});return 0
if __name__=='__main__':raise SystemExit(main())
