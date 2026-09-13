"""Bounded Yunwu client. No automatic media loading, retries or credential logs."""
import json,os
from pathlib import Path
import requests
BASE_URL='https://yunwu.ai/v1'
DEFAULT_MODEL='deepseek-v4-flash'
CREDENTIAL_FILE=Path.home()/'.config/react2025/yunwu.key'

def api_key():
    key=os.environ.get('YUNWU_API_KEY')
    if not key and CREDENTIAL_FILE.exists():
        if CREDENTIAL_FILE.stat().st_mode & 0o077:raise RuntimeError('Credential file must be private (0600)')
        key=CREDENTIAL_FILE.read_text().strip()
    if not key:raise RuntimeError('Yunwu credential is not configured')
    return key

def chat(messages,*,model=DEFAULT_MODEL,max_tokens=200,timeout=60,thinking=None):
    if not 1<=max_tokens<=4096:raise ValueError('Explicit bounded token limit required')
    # requests does not retry POST by default. Do not log headers, request media,
    # credentials or raw provider errors, which can contain request echoes.
    payload={'model':model,'messages':messages,'max_tokens':max_tokens}
    if thinking is None and model=='deepseek-v4-flash':thinking={'type':'disabled'}
    if thinking is not None:payload['thinking']=thinking
    response=requests.post(BASE_URL+'/chat/completions',headers={'Authorization':'Bearer '+api_key(),'Content-Type':'application/json'},json=payload,timeout=timeout)
    if response.status_code!=200:raise RuntimeError(f'Yunwu HTTP {response.status_code}; response body suppressed')
    data=response.json()
    if not isinstance(data.get('choices'),list) or not data['choices']:raise RuntimeError('Yunwu response has no choices')
    return data

def healthcheck():
    result=chat([{'role':'user','content':'Reply with the single word OK.'}],max_tokens=16)
    content=result['choices'][0]['message'].get('content','')
    # Store only whether the innocuous expected response was obtained.
    return dict(provider='Yunwu',base_url=BASE_URL,requested_model=DEFAULT_MODEL,returned_model=result.get('model'),http_status=200,responded_ok=isinstance(content,str) and content.strip()=='OK',usage=result.get('usage'),research_data_sent=False,media_sent=False,requests=1)
