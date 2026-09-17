"""Read-only raw-record audit. No interpolation, resampling, training joins or model fitting."""
import collections
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
DATA = OUT.parents[2]


def numeric(s):
    x = pd.to_numeric(s, errors='coerce')
    finite = x[np.isfinite(x)]
    return dict(n=len(s), missing=int(x.isna().sum()), negative=int((x < 0).sum()),
                zero=int((x == 0).sum()), nonfinite=int((x.notna() & ~np.isfinite(x)).sum()),
                min=float(finite.min()) if len(finite) else None,
                p50=float(finite.median()) if len(finite) else None,
                p99=float(finite.quantile(.99)) if len(finite) else None,
                max=float(finite.max()) if len(finite) else None)


def times(s, value=None, cgm=False):
    t = pd.to_datetime(s, dayfirst=True, format='mixed', errors='coerce')
    a = t.dropna().sort_values().drop_duplicates()
    gaps = a.diff().dt.total_seconds().dropna() / 60
    ans = dict(rows=len(s), invalid=int(t.isna().sum()), duplicate_times=int(t.duplicated().sum()),
               start=str(a.min()) if len(a) else None, end=str(a.max()) if len(a) else None,
               unique_times=len(a), span_days=float((a.max()-a.min()).total_seconds()/86400) if len(a) else 0,
               native_order_backwards=int((t.diff().dt.total_seconds() < 0).sum()),
               gap_minutes=numeric(gaps), gaps_gt_7_5m=int((gaps > 7.5).sum()),
               gaps_gt_30m=int((gaps > 30).sum()), gaps_gt_360m=int((gaps > 360).sum()))
    if value is not None:
        f=pd.DataFrame({'t': t, 'v': value})
        ans['same_time_conflicting_values'] = int((f.dropna(subset=['t']).groupby('t')['v'].nunique(dropna=False)>1).sum())
    if cgm:
        # Distinct observed timestamp runs; no new samples are generated.
        run=(a.diff().dt.total_seconds().fillna(0)>450).cumsum()
        spans=a.groupby(run).agg(['min','max'])
        hours=(spans['max']-spans['min']).dt.total_seconds()/3600
        ans['observed_runs_gap_le_7_5m'] = {'count':len(hours), 'max_hours':float(hours.max()) if len(hours) else 0,
             'runs_ge_6h':int((hours>=6).sum()),'hours_in_runs_ge_6h':float(hours[hours>=6].sum()),
             'note':'CGM continuity only; includes no insulin/meal eligibility or history warm-up requirement'}
    return ans


def audit_ohio():
    records=[]; pools=collections.defaultdict(list); attrs={}
    for path in sorted((DATA/'extracted/ohio').rglob('*.xml')):
        root=ET.parse(path).getroot(); person=root.attrib['id'];attrs[person]=dict(root.attrib)
        file={'file':str(path.relative_to(DATA/'extracted/ohio')),'subject':person,'split':path.parent.name,'sections':{}}
        for elem in root:
            rows=[dict(e.attrib) for e in elem];f=pd.DataFrame(rows)
            item={'rows':len(rows),'columns':list(f.columns),'missing_by_column':{k:int(v) for k,v in f.isna().sum().items()},
                  'exact_duplicate_rows':int(f.duplicated().sum())}
            for col in ['value','dose','carbs','duration','quality','intensity']:
                if col in f:item[col]=numeric(f[col])
            for col in ['ts','ts_begin','ts_end','tbegin','tend']:
                if col in f:item[col]=times(f[col],f['value'] if 'value' in f else None,elem.tag=='glucose_level' and col=='ts')
            if 'type' in f:item['types']=f['type'].value_counts(dropna=False).to_dict()
            if {'ts_begin','ts_end'}<=set(f):
                start=pd.to_datetime(f.ts_begin,dayfirst=True,format='mixed',errors='coerce');end=pd.to_datetime(f.ts_end,dayfirst=True,format='mixed',errors='coerce')
                item['interval_minutes']=numeric((end-start).dt.total_seconds()/60)
                item['invalid_interval_timestamps']=int((start.isna()|end.isna()).sum())
            file['sections'][elem.tag]=item
            if elem.tag in ['glucose_level','basal','temp_basal','bolus','meal']:
                f['_file_split']=path.parent.name;pools[(person,elem.tag)].append(f)
        records.append(file)
    people=[]
    for person in sorted(attrs):
        row={'subject':person,'xml_attributes':attrs[person],'sections':{}}
        for tag in ['glucose_level','basal','temp_basal','bolus','meal']:
            f=pd.concat(pools[(person,tag)],ignore_index=True);tcol='ts' if 'ts' in f else 'ts_begin'
            row['sections'][tag]={'records':len(f),'time':times(f[tcol],f['value'] if 'value' in f else None,tag=='glucose_level'),
                                   'exact_duplicates_ignoring_file_split':int(f.drop(columns='_file_split').duplicated().sum())}
        f=pd.concat(pools[(person,'glucose_level')],ignore_index=True)
        splitsets={k:set(v.ts) for k,v in f.groupby('_file_split')}
        row['train_test_same_timestamp_count']=len(splitsets.get('train',set())&splitsets.get('test',set()))
        people.append(row)
    totals=collections.Counter()
    for f in records:
        for k,v in f['sections'].items():totals[k]+=v['rows']
    return {'method':'All 24 XML parsed, raw fields only. Per-person chronological union for diagnostic continuity, never a training split or aligned table.',
            'files':records,'patients':people,'totals':dict(totals),'n_patients':len(people)}


def audit_uom():
    root=next((DATA/'extracted/uom').iterdir());files=[]
    for path in sorted(root.rglob('*.csv')):
        f=pd.read_csv(path); ids=re.findall(r'(?:23|24)\d{2}',path.stem)
        person=ids[-1] if ids else None
        category=str(path.parent.relative_to(root))
        item={'file':str(path.relative_to(root)),'subject':person,'category':category,'rows':len(f),
              'columns':list(f.columns),'missing':{k:int(v) for k,v in f.isna().sum().items()},'exact_duplicate_rows':int(f.duplicated().sum()),
              'numeric':{c:numeric(f[c]) for c in f if pd.api.types.is_numeric_dtype(f[c])}}
        tc=next((c for c in f if c.endswith('_ts')),None)
        if tc:
            vc='value' if 'value' in f else 'basal_dose' if 'basal_dose' in f else 'bolus_dose' if 'bolus_dose' in f else None
            item['time']=times(f[tc],f[vc] if vc else None,category=='Glucose Data')
            # Format evidence, not a datetime transformation in persisted patient data.
            ss=f[tc].astype(str);item['first_component_gt12']=int((pd.to_numeric(ss.str.extract(r'^(\d{1,2})/')[0],errors='coerce')>12).sum())
        if 'insulin_kind' in f:item['insulin_kind']=f.insulin_kind.value_counts(dropna=False).to_dict()
        if 'bolus_dose' in f:item['bolus_unique_values']=int(f.bolus_dose.nunique())
        files.append(item)
    categories={c:{'files':sum(f['category']==c for f in files),'rows':sum(f['rows'] for f in files if f['category']==c),
                    'subjects':sorted({f['subject'] for f in files if f['category']==c and f['subject']})} for c in {f['category'] for f in files}}
    people=[]
    for person in sorted({f['subject'] for f in files if f['category']=='Glucose Data'}):
        fs=[f for f in files if f['subject']==person];by={f['category']:f for f in fs if f['category']!='Sleep Data'}
        row={'subject':person,'available_categories':sorted({f['category'] for f in fs}),
             'basal_kinds':by.get('Insulin Data/Basal Data',{}).get('insulin_kind',{})}
        # Common observed support endpoints only; this is an upper bound, not actual paired coverage.
        required=['Glucose Data','Insulin Data/Basal Data','Insulin Data/Bolus Data']
        if all(c in by for c in required):
            ranges=[by[c]['time'] for c in required];lo=max(pd.Timestamp(r['start']) for r in ranges);hi=min(pd.Timestamp(r['end']) for r in ranges)
            row['common_record_extent_days_upper_bound']=max(0.,(hi-lo).total_seconds()/86400)
        people.append(row)
    return {'method':'Full scan of all CSV. Dates parsed dayfirst for diagnostics based on actual day>12 strings; original CSV unchanged. Common extent is not alignment/coverage.',
            'files':files,'categories':categories,'patients':people}


if __name__=='__main__':
    for name,fun in [('ohio',audit_ohio),('uom',audit_uom)]:
        result=fun();(OUT/(name+'_audit.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        print(name, json.dumps(result.get('totals',result.get('categories')),ensure_ascii=False))
