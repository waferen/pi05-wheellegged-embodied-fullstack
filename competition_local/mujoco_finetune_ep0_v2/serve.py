"""Serve the completed pilot with its own training model and norm statistics."""
import argparse,json,logging,os
from experiment import config,HERE,ROOT
os.environ.setdefault('OPENPI_DATA_HOME',str(ROOT/'cloud_snapshot/cache/openpi'))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR','platform')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8767);args=parser.parse_args()
    cfg=config();assert json.loads((HERE/'train_result.json').read_text())['passed']
    paths=sorted((p for p in cfg.checkpoint_dir.iterdir() if p.is_dir() and p.name.isdigit()),key=lambda p:int(p.name))
    checkpoint=paths[-1]
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    policy=create_trained_policy(cfg,checkpoint)
    print('Serving trained pilot',checkpoint,flush=True)
    (HERE/'serve_checkpoint.json').write_text(json.dumps({'checkpoint':str(checkpoint),'port':args.port}))
    WebsocketPolicyServer(policy,host='127.0.0.1',port=args.port,metadata=policy.metadata).serve_forever()


if __name__=='__main__':logging.basicConfig(level=logging.INFO);main()
