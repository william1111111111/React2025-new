"""Versioned proposal/annotation schemas; UNKNOWN is never a negative label."""
from .build import ANN, write

def obj(properties,required=None):return dict(type='object',properties=properties,required=list(properties) if required is None else required,additionalProperties=False)
S={'type':'string'}; B={'type':'boolean'}; N={'type':['number','null']}; I={'type':['integer','null'],'minimum':0}; nullable_list={'type':['array','null'],'items':S}
common=dict(schema_version={'const':'reaction-program-annotation-v1'},split={'enum':['train','val','test']},fold=S)
span={'type':'array','items':{'type':'integer','minimum':0},'minItems':2,'maxItems':2}
confidence={'type':['number','null'],'minimum':0,'maximum':1}
unknown_meta=dict(confidence=confidence,confidence_calibrated={'const':False},human_reviewed=B,training_eligible=B)
action=obj(dict(description=S,region={'type':['string','null']},onset=I,peak=I,offset=I,intensity=obj(dict(value=N,scale={'enum':['ordinal_visible','numeric_from_dataset','unavailable']},evidence=S)),evidence_frame_ids={'type':'array','items':{'type':'integer','minimum':0}},uncertainty_reasons={'type':'array','items':S}))
schemas={
'speaker_events':obj(dict(**common,clip_id=S,session_id=S,event_id=S,status={'enum':['candidate','auto_weak','reviewed']},start_frame=I,end_frame=I,anchor_frame=I,transcript=S,transcript_span=span,event_type={'type':'array','items':S},prosody=nullable_list,visible_behavior=nullable_list,semantic_summary=S,role_status={'enum':['UNKNOWN','unknown','source_speaker_candidate','source_supported','other_voice']},time_domain=S,timing_evidence={'type':['object','null']},sync_evidence={'type':['object','null']},**unknown_meta,visible_roles={'const':['speaker']},provenance={'type':'object'})),
'listener_reactions':obj(dict(**common,clip_id=S,observation_id=S,event_id={'type':['string','null']},status={'enum':['pending_observation','observed','UNKNOWN']},reaction_start=I,reaction_peak=I,reaction_end=I,actions={'type':['array','null'],'items':action},global_style={'type':['object','null']},**unknown_meta,visible_roles={'const':['listener']},numeric_assets={'type':'object','additionalProperties':S},video_asset=S,linked_candidate_ids={'type':'array','items':S},relation_status={'enum':['UNKNOWN','temporal_support']},time_domain={'const':'listener_native_pts'},provenance={'type':'object'})),
'event_equivalence':obj(dict(**common,event_a=S,event_b=S,equivalence=confidence,status={'enum':['UNKNOWN','compatible','incompatible']},reason=obj({k:{'type':['boolean','null']} for k in ['same_intent','same_dialogue_stage','compatible_local_context']}),visible_roles={'const':['speaker']},confidence_calibrated={'const':False},human_reviewed=B,training_eligible=B,provenance={'type':'object'})),
'reaction_support':obj(dict(**common,event_id=S,observed_support={'type':'array','items':obj(dict(observation_id=S,program_id=S,association_evidence=S,equivalence_evidence={'type':['string','null']}))},LLM_inferred_support={'type':'array','items':{'type':'object'}},status={'enum':['UNKNOWN','supported']},training_eligible=B,missing_is_not_no_reaction={'const':True})),
'raw_response_receipt':obj(dict(task_id=S,split={'enum':['train','val','test']},request_sha256=S,input_asset_hashes={'type':'array','items':S},prompt_sha256=S,requested_model=S,returned_model={'type':['string','null']},model_revision={'type':['string','null']},revision_status={'enum':['provider_supplied','unknown']},raw_response_path=S,raw_response_sha256=S,accessed_modalities={'type':'array','items':S},schema_errors={'type':'array','items':S},human_reviewed={'const':False},confidence_calibrated={'const':False}))}

def main():
    for name,schema in schemas.items():
        schema={'$schema':'https://json-schema.org/draft/2020-12/schema','$id':'urn:react:reaction-program-v1:'+name,**schema}
        write(ANN/'schema'/(name+'.schema.json'),schema)
if __name__=='__main__':main()
