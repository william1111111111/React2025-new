from .common import *
def main():
 audit=read(ROOT/'SPLIT_CONTENT_AUDIT.json');manifest=read(ROOT/'manifest.json');combined={};lines=['# Complete local TEST evaluation','','Local TEST is NOT an independent holdout: all 3426 audited input/target feature files are byte-identical to VAL. Checkpoints were fixed before this run; results do not tune training.','','Official directional population; original speaker and reverse directions reported separately. K10 native source-only; all TEST semantic inputs NULL because no split-legal TEST cache exists. Full exact FRD, not FRD20.','','| Model / group | N | FRC | exact FRD | S-MSE | FRVar | temporal S-MSE | TLCC | MAE |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for a in ARMS:
  m=read(ROOT/(a+'_RESULTS.json'));f=read(ROOT/(a+'_FRD_RESULTS.json'));assert f['completed'] and f['pairs']==manifest['population']*100;combined[a]=dict(checkpoint=m['checkpoint'],groups={})
  for group,metrics in m['results'].items():
   values={**metrics,'FRD':f['results'][group]['FRD']};combined[a]['groups'][group]=values
   lines.append('| '+a+' / '+group+' | '+str(values['count'])+' | '+' | '.join(f'{values[k]:.9f}' for k in ['FRC','FRD','S_MSE','FRVar','temporal_S_MSE','TLCC','MAE'])+' |')
 write(ROOT/'COMPLETE_RESULTS.json',dict(split_audit_summary={k:v for k,v in audit.items() if k!='rows'},models=combined));(ROOT/'COMPLETE_RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
