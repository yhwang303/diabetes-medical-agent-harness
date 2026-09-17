from pathlib import Path
import duckdb,json,time,csv,hashlib,shutil
import pandas as pd,numpy as np,collections,pyarrow as pa,pyarrow.parquet as pq
P=Path(__file__).resolve().parents[1];SRC=P.parent/'原始数据/Data Tables';TMP=P/'内部';TMP.mkdir(exist_ok=True)
c=duckdb.connect(str(TMP/'metadata.duckdb'));c.execute("SET memory_limit='12GB'; SET threads=2; SET partitioned_write_flush_threshold=8192; SET partitioned_write_max_open_files=24; SET preserve_insertion_order=false")
c.execute('SET temp_directory=?',[str(TMP/'spill')])
def q(s):return "'"+str(s).replace("'","''")+"'"
def source(f):
 header=f.open('rb').readline().decode().strip().split('|')
 cols='{'+','.join(q(k)+":'VARCHAR'" for k in header)+'}'
 return f"read_csv({q(f)},auto_detect=false,header=true,delim='|',quote='\"',escape='\"',columns={cols},null_padding=true,strict_mode=false,max_line_size=10000000)"
# Small tables: validate identity and normalize only timestamps/types, retaining all source values.
for table in ['PtRoster','Surveys','LOOPDeviceUploads','LOOPPtFinalStatus','SampleResults','gluIndices','adverseEvents','deviceDiscontSurvey','deviceIssues','LOOPContactInteraction']:
 if c.execute('select count(*) from information_schema.tables where table_name=?',[table]).fetchone()[0]:continue
 f=SRC/(table+'.txt');c.execute(f'CREATE TABLE "{table}" AS SELECT * FROM {source(f)}');print('META',table,c.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0],flush=True)
assert c.execute('select count(*)=count(distinct PtID) from PtRoster').fetchone()[0]
assert c.execute('select count(*)=count(distinct RecID) from LOOPDeviceUploads').fetchone()[0]
# All device files, partitioned on explicitly parsed patient identity. No record is dropped.
files=[f for f in sorted(SRC.glob('LOOPDevice*.txt')) if f.stem not in ['LOOPDeviceUploads','LOOPDeviceIssueRpt']]
manifestpath=P/'审计/ingestion.json';M=json.loads(manifestpath.read_text()) if manifestpath.exists() else {}
for f in files:
 if f.name in M:continue
 kind=''.join(x for x in f.stem.removeprefix('LOOPDevice') if not x.isdigit()).lower()
 out=TMP/'按患者原事件'/kind/f.stem
 if out.exists():shutil.rmtree(out) # incomplete generated partition only; originals never touched
 out.mkdir(parents=True,exist_ok=True)
 # Bounded pandas chunks avoid the large CSV join materialization observed in DuckDB.
 up=c.execute('SELECT RecID,PtID,DataSource FROM LOOPDeviceUploads').fetchdf().set_index('RecID')
 patient_ids=set(c.execute('select PtID from PtRoster').fetchdf().PtID)
 n=0;statuses=collections.Counter();badtime=0
 for j,d in enumerate(pd.read_csv(f,sep='|',dtype=str,chunksize=200000)):
  d['pid']=pd.to_numeric(d.PtID,errors='raise').astype('int64');d['source_file']=f.name;d['source_record_number']=np.arange(n+2,n+2+len(d));n+=len(d)
  d['event_time']=pd.to_datetime(d.UTCDtTm,format='%Y-%m-%d %H:%M:%S',errors='coerce')
  parent=d.ParentLOOPDeviceUploadsID.map(up.PtID);d['upload_source']=d.ParentLOOPDeviceUploadsID.map(up.DataSource)
  d['identity_status']=np.select([~d.PtID.isin(patient_ids),parent.isna(),parent.ne(d.PtID)],['unknown_patient','unknown_upload','patient_upload_mismatch'],default='ok')
  statuses.update(d.identity_status.value_counts().to_dict());badtime+=int(d.event_time.isna().sum())
  for pid,g in d.groupby('pid',sort=False):
   folder=out/f'pid={pid}';folder.mkdir(exist_ok=True);g.drop(columns='pid').to_parquet(folder/f'part_{j:05d}.parquet',index=False,compression='zstd')
  if j%10==0:print('CHUNK',f.name,n,flush=True)
 M[f.name]={'rows':n,'kind':kind,'identity_status_counts':dict(statuses),'invalid_event_time':badtime}
 manifestpath.write_text(json.dumps(M,indent=2));print('INGESTED',f.name,n,dict(statuses),flush=True)
# IssueRpt has malformed complex text. Preserve byte-exact original; record each anomalous logical row separately.
f=SRC/'LOOPDeviceIssueRpt.txt';rows=[];bad=[]
with f.open(encoding='latin1',newline='') as h:
 reader=csv.reader(h,delimiter='|');header=next(reader)
 for i,row in enumerate(reader,2):
  if len(row)!=len(header):bad.append({'logical_record':i,'width':len(row),'values_latin1_lossless':row})
  else:rows.append(dict(zip(header,row)))
(P/'隔离记录').mkdir(exist_ok=True)
(P/'隔离记录/IssueRpt结构异常.json').write_text(json.dumps(bad,ensure_ascii=False));(TMP/'IssueRpt_valid.json').write_text(json.dumps(rows,ensure_ascii=False))
print('DONE',sum(x['rows'] for x in M.values()),'IssueRpt malformed',len(bad),flush=True)
