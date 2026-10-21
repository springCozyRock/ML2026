"""Kaggle API adapter. Pass KAGGLE_API_TOKEN in the process environment.

Works with the server's official kaggle 1.7.4.5 SDK. No credentials are persisted.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import requests

COMPETITION='house-prices-advanced-regression-techniques'
ROOT=Path(__file__).resolve().parent

def list_submissions():
    r=requests.get('https://www.kaggle.com/api/v1/competitions/submissions/list/'+COMPETITION,
                   headers={'Authorization':'Bearer '+os.environ['KAGGLE_API_TOKEN']},timeout=40)
    r.raise_for_status()
    # Retain only report-relevant, non-credential fields.
    entries=[{k:s.get(k) for k in ['ref','date','description','fileName','publicScore',
                                 'privateScore','status','errorDescription']} for s in r.json()]
    dest=ROOT/'output';dest.mkdir(exist_ok=True)
    (dest/'kaggle_submissions.json').write_text(json.dumps(entries,indent=2))
    print(json.dumps(entries,indent=2))
    return entries

def submit(path,description):
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk import KaggleClient
    @contextmanager
    def client():
        with KaggleClient(username='',password='') as c:
            c._http_client._session.auth=None
            c._http_client._session.headers['Authorization']='Bearer '+os.environ['KAGGLE_API_TOKEN']
            send=c._http_client._session.send
            def timed_send(request,**kwargs):
                kwargs['timeout']=40
                print('Kaggle API request',request.method,request.url.split('?')[0],flush=True)
                return send(request,**kwargs)
            c._http_client._session.send=timed_send
            yield c
    api=KaggleApi()
    api.build_kaggle_client=client
    upload=api.upload_complete
    def timed_upload(*args,**kwargs):
        print('Submission upload started',flush=True)
        result=upload(*args,**kwargs)
        print('Submission upload finished',flush=True)
        return result
    api.upload_complete=timed_upload
    result=api.competition_submit(str(path),description,COMPETITION,quiet=True)
    # SDK response might contain upload capabilities. Print only submission status/id.
    print('submission response', {k:getattr(result,k,None) for k in ['ref','submission_id','message']})
    list_submissions()

def submit_http(path,description):
    """Same official endpoints and fields as kaggle SDK, with bounded timeouts."""
    path=Path(path);headers={'Authorization':'Bearer '+os.environ['KAGGLE_API_TOKEN']}
    r=requests.post('https://www.kaggle.com/api/v1/competitions/submission-url',headers=headers,
                    files={k:(None,str(v)) for k,v in {'competitionName':COMPETITION,'contentLength':path.stat().st_size,
                          'lastModifiedEpochSeconds':int(path.stat().st_mtime),'fileName':path.name}.items()},timeout=40)
    print('Upload initialization status',r.status_code,flush=True)
    if not r.ok:raise RuntimeError('Upload initialization rejected: HTTP '+str(r.status_code))
    data=r.json()
    # Signed upload URL and file token are never logged.
    url=data.get('createUrl') or data.get('create_url')
    token=data.get('token')
    if not url or not token:raise RuntimeError('Unexpected upload schema: '+str(list(data)))
    with path.open('rb') as f:
        upload=requests.put(url,data=f,timeout=60)
    print('File upload status',upload.status_code,flush=True)
    if not upload.ok:raise RuntimeError('File upload failed: HTTP '+str(upload.status_code))
    r=requests.post('https://www.kaggle.com/api/v1/competitions/submissions/submit/'+COMPETITION,
                    headers=headers,files={k:(None,str(v)) for k,v in {'competitionName':COMPETITION,'blobFileTokens':token,
                                          'submissionDescription':description}.items()},timeout=40)
    print('Create submission status',r.status_code,flush=True)
    if not r.ok:raise RuntimeError('Create submission failed: HTTP '+str(r.status_code))
    result=r.json();print({k:v for k,v in result.items() if k in ['ref','message','submissionId']},flush=True)
    list_submissions()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['list','submit','submit-http']);p.add_argument('--file');p.add_argument('--description',default='Reproducible House Prices experiment')
    a=p.parse_args()
    if a.action=='list':list_submissions()
    elif a.action=='submit':submit(a.file,a.description)
    else:submit_http(a.file,a.description)
