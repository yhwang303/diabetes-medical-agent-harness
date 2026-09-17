from audit_trials import *
p=OUT/'dclp3_audit.json';r=json.loads(p.read_text());d=read(ROOT/'datasets/extracted/dclp3/Data Files/cgm.txt',usecols=['PtID','DataDtTm']);r['tables']['cgm.txt']['raw_time']=time_stats(d,'PtID','DataDtTm');dump('dclp3_audit.json',r)
