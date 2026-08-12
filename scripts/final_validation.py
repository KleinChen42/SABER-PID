"""Deterministic pre-submission assertions over final artifacts."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def read(path): return json.loads(path.read_text(encoding="utf-8"))
def main() -> int:
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();root=Path(a.root).resolve();checks=[]
    f2=read(root/'reports/generated/qwen8_selection_prompt_resolution_matrix.json');checks.append(('f2_cells',f2.get('cell_count')==12));checks.append(('f2_records',all(v==400 for v in f2.get('records_per_cell',{}).values())));checks.append(('f2_missing',all(c.get('missing_prediction_count')==0 for c in f2.get('cells',{}).values())))
    f3=read(root/'reports/generated/cross_family_resolution_bootstrap.json');checks.append(('f3_cells',len(f3.get('cells',{}))==4));checks.append(('f3_missing',all(c.get('missing_prediction_count')==0 for c in f3.get('cells',{}).values())))
    f5=read(root/'reports/generated/degradation_severity_analysis_v2.json');checks.append(('f5_conditions',len(f5.get('rows',[]))==9));checks.append(('f5_missing',all(x.get('missing_prediction_count')==0 for x in f5.get('rows',[]))))
    f6=read(root/'reports/generated/efficiency_measurement_audit_v2.json').get('audit',{});checks.append(('f6_status',f6.get('status')=='pass'));checks.append(('f6_rows',f6.get('actual_measurement_rows')==2400));checks.append(('f6_duplicates',f6.get('duplicate_measurement_rows')==0))
    rel=read(root/'reports/generated/final_release_manifest.json');checks.append(('release_status',rel.get('status')=='pass'));checks.append(('release_missing',not rel.get('missing_artifacts')))
    num=read(root/'reports/generated/final_manuscript_number_audit.json');checks.append(('number_audit',num.get('status')=='pass' and num.get('all_present')))
    result={'status':'pass' if all(v for _,v in checks) else 'fail','checks':dict(checks)};print(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True));return 0 if result['status']=='pass' else 1
if __name__=='__main__':raise SystemExit(main())
