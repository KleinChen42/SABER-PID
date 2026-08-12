"""Add late-created submission artifacts to the final registry."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();root=Path(a.root).resolve();path=root/'reports/generated/final_experiment_registry.json';d=json.loads(path.read_text(encoding='utf-8'));extras=['outputs/telemetry/efficiency_repeats_v2.jsonl','data/manifests/pid2graph_open100_v1.json','reports/generated/open100_external_resolution_table.csv','reports/generated/open100_external_bootstrap.json','reports/F6_EFFICIENCY_EXECUTION_V1.md','reports/TEST_VALIDATION_V1.md','reports/GIT_RELEASE_STATUS_V1.md','paper/title_page.md','paper/declarations.md','paper/figure_manifest.md'];items={x['path']:x for x in d.get('artifacts',[])}
 for rel in extras:
  pth=root/rel
  if pth.exists():items[rel]={'path':rel,'exists':True,'bytes':pth.stat().st_size,'sha256':hashlib.sha256(pth.read_bytes()).hexdigest()}
 d['artifacts']=sorted(items.values(),key=lambda x:x['path']);d['artifact_count']=len(d['artifacts']);path.write_text(json.dumps(d,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');print({'status':d.get('status'),'artifact_count':d['artifact_count']});return 0
if __name__=='__main__':raise SystemExit(main())
