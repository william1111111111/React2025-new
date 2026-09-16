"""Nearest real PTS with explicit support, residual and missing-frame checks."""
import numpy as np

def nearest(pts,indices,shift,nominal_interval,precision=1e-6):
 pts=np.asarray(pts,dtype=np.float64);indices=np.asarray(indices,dtype=int)
 if len(pts)<2 or not np.all(np.diff(pts)>0):raise ValueError('PTS must strictly increase')
 queries=pts[indices]+shift;support=(queries>=pts[0])&(queries<=pts[-1]);ii=indices[support];q=queries[support]
 # Clipping is only for valid in-support neighbor lookup, never to rescue a query.
 right=np.searchsorted(pts,q);right=np.minimum(right,len(pts)-1);left=np.maximum(right-1,0)
 jj=np.where(np.abs(pts[left]-q)<=np.abs(pts[right]-q),left,right)
 residual=pts[jj]-q
 prev=np.maximum(jj-1,0);nxt=np.minimum(jj+1,len(pts)-1)
 local=np.minimum(np.where(jj>0,pts[jj]-pts[prev],np.inf),np.where(jj<len(pts)-1,pts[nxt]-pts[jj],np.inf))
 tolerance=.5*np.minimum(local,nominal_interval)+precision
 checks=dict(supported_min60=len(ii)>=60,source_contiguous=bool(len(ii)>0 and np.all(np.diff(ii)==1)),mapped_contiguous=bool(len(jj)>0 and np.all(np.diff(jj)==1)),residual_within_limit=bool(len(jj)>0 and np.all(np.abs(residual)<=tolerance)),source_no_large_gap=bool(len(ii)>1 and np.all(np.diff(pts[ii])<=1.5*nominal_interval+precision)),mapped_no_large_gap=bool(len(jj)>1 and np.all(np.diff(pts[jj])<=1.5*nominal_interval+precision)))
 diag=dict(expected_frames=len(indices),supported_frames=len(ii),outside_support=int((~support).sum()),original_support=[float(pts[0]),float(pts[-1])],query_range=[float(queries.min()),float(queries.max())],source_indices=ii.tolist(),matched_indices=jj.tolist(),signed_residual_s=residual.tolist(),allowed_residual_s=tolerance.tolist(),max_abs_residual_s=float(np.abs(residual).max()) if len(residual) else None,checks=checks)
 return ii,jj,diag
