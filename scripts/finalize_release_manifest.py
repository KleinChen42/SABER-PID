"""Remove the inherently self-referential release-manifest hash entry."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> int:
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();root=Path(a.root).resolve();path=root/'reports/generated/final_release_manifest.json';data=json.loads(path.read_text(encoding='utf-8'));data['items']=[x for x in data.get('items',[]) if x.get('path')!='reports/generated/final_release_manifest.json'];data['artifact_count']=len(data['items']);data['self_hash_policy']='self entry omitted because a manifest cannot contain its own stable hash';path.write_text(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');print(json.dumps({'status':data.get('status'),'artifact_count':data['artifact_count'],'missing_artifacts':data.get('missing_artifacts',[])},indent=2,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
