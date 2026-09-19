"""Upload and inference service behind an SSH or HTTPS reverse tunnel."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import uuid
from inference_jobs import execute,TIERS


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--port',type=int,default=8787)
    a=p.parse_args();config=json.loads(a.config.read_text());root=Path(config['jobs']);root.mkdir(parents=True,exist_ok=True)
    static=Path(__file__).parent/'static';pool=ThreadPoolExecutor(max_workers=1)
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,data,kind='application/json'):
            body=json.dumps(data,ensure_ascii=False).encode() if kind=='application/json' else data
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        def body(self,limit):
            size=int(self.headers.get('Content-Length','0'))
            if size<=0 or size>limit: raise ValueError('File size must be between 1 byte and 128 MB')
            return self.rfile.read(size)
        def job(self):
            parts=self.path.split('/')
            if len(parts)<4 or not re.fullmatch('[a-f0-9]{32}',parts[3]): raise ValueError('Invalid job identifier')
            job=root/parts[3]
            if not (job/'status.json').exists(): raise ValueError('Job does not exist')
            return job
        def do_GET(self):
            if self.path=='/api/model':return self.send(200,{k:config[k] for k in ('model_label','model_status')})
            if self.path.startswith('/api/jobs/'):
                try:
                    job=self.job();state=json.loads((job/'status.json').read_text())
                    if self.path.endswith('/result'):
                        if state['status']!='completed':return self.send(409,{'error':'Result is pending'})
                        return self.send(200,(job/'result.json').read_bytes(),'application/json; charset=utf-8')
                    if self.path.endswith('/download'):
                        if state['status']!='completed':return self.send(409,{'error':'Result is pending'})
                        return self.send(200,(job/'output'/state['output_filename']).read_bytes(),'application/octet-stream')
                    return self.send(200,state)
                except ValueError as e:return self.send(400,{'error':str(e)})
            names={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
            page=self.path.split('?',1)[0]
            if page not in names:return self.send(404,{'error':'Not found'})
            f=static/names[page];return self.send(200,f.read_bytes(),mimetypes.guess_type(f.name)[0]+'; charset=utf-8')
        def do_PUT(self):
            try:
                job=self.job();name=self.path.rsplit('/',1)[-1]
                if name not in ('topology.pdb','observations.xtc'):raise ValueError('Expected PDB or XTC file')
                if json.loads((job/'status.json').read_text())['status']!='uploading':raise ValueError('Job already submitted')
                (job/name).write_bytes(self.body(128*1024*1024));return self.send(200,{'uploaded':name})
            except ValueError as e:return self.send(400,{'error':str(e)})
        def do_POST(self):
            try:
                if self.path=='/api/jobs':
                    data=json.loads(self.body(4096));tier=data['tier'];ligand=data['ligand'].strip().upper()
                    if tier not in TIERS or not re.fullmatch('[A-Z0-9]{1,4}',ligand):raise ValueError('Invalid tier or ligand residue name')
                    identifier=uuid.uuid4().hex;job=root/identifier;job.mkdir()
                    state=dict(id=identifier,tier=tier,ligand=ligand,status='uploading');(job/'status.json').write_text(json.dumps(state))
                    return self.send(201,state)
                job=self.job()
                if not self.path.endswith('/run'):return self.send(404,{'error':'Not found'})
                state=json.loads((job/'status.json').read_text())
                if state['status']!='uploading':raise ValueError('Job already submitted')
                if not all((job/f).is_file() for f in ('topology.pdb','observations.xtc')):raise ValueError('Upload both files')
                state['status']='queued';(job/'status.json').write_text(json.dumps(state));pool.submit(execute,job,config)
                return self.send(202,state)
            except (ValueError,KeyError) as e:return self.send(400,{'error':str(e)})
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    print(f'QField-Dyn inference service on 127.0.0.1:{a.port}',flush=True);server.serve_forever()


if __name__=='__main__':main()
