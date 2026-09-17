from audit_trials import *
base=ROOT/'datasets/extracted/iobp2/Data Tables';roster=read(base/'IOBP2PtRoster.txt').set_index('PtID');parts=[]
cols=['PtID','UploadIndex','DeviceDtTm','CGMVal','InsDelivPrev','BasalDelivPrev','BolusDelivPrev','MealBolusDelivPrev','InsDelivAvail']
for d in read(base/'IOBP2DeviceiLet.txt',usecols=cols,chunksize=400000):
 d['component_residual']=d.InsDelivPrev-d.BasalDelivPrev-d.BolusDelivPrev-d.MealBolusDelivPrev;parts.append(d[d.component_residual.abs()>.00200001])
x=pd.concat(parts);ad=x.PtID.map(roster.AgeAsofEnrollDt)>=18
out={'rows':len(x),'patients':x.PtID.nunique(),'adult_rows':int(ad.sum()),'adult_patients':x.loc[ad,'PtID'].nunique(),'adult_groups':counts(x.loc[ad,'PtID'].map(roster.TrtGroup)),'residual_quantiles':quant(x.component_residual),'channel_available':counts(x.InsDelivAvail),'upload_index_quantiles':quant(x.UploadIndex),'all_components_zero_rows':int(((x.BasalDelivPrev==0)&(x.BolusDelivPrev==0)&(x.MealBolusDelivPrev==0)).sum()),'total_zero_rows':int((x.InsDelivPrev==0).sum())};dump('iobp2_component_exceptions.json',out)
