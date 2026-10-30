"""Serve the migrated 40000-step B2W policy from the existing WSL environment."""
import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / 'cloud_snapshot'
CHECKPOINT = SNAPSHOT / 'openpi/checkpoints/pi05_b2w_piper/run01/40000'
VERIFICATION = ROOT / 'migration/checkpoint_40000_20261028/verification.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--preflight', action='store_true', help='Check local paths and transforms without loading weights')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')
    os.environ.setdefault('OPENPI_DATA_HOME', str(SNAPSHOT / 'cache/openpi'))
    from openpi.training import config

    base = config.get_config('pi05_b2w_piper')
    data = dataclasses.replace(
        base.data,
        assets=dataclasses.replace(base.data.assets, assets_dir=str(SNAPSHOT / 'openpi/assets/pi05_b2w_piper')),
        base_config=dataclasses.replace(base.data.base_config, local_root=str(SNAPSHOT / 'lerobot_datasets/b2w_piper_c4_transport')),
    )
    cfg = dataclasses.replace(base, data=data, assets_base_dir=str(SNAPSHOT / 'openpi/assets'))
    d = cfg.data.create(cfg.assets_dirs, cfg.model)
    report = {
        'checkpoint': str(CHECKPOINT),
        'config': cfg.name,
        'model_unchanged': cfg.model == base.model,
        'action_horizon': cfg.model.action_horizon,
        'normalization_asset_id': d.asset_id,
        'assets_dir': str(cfg.assets_dirs),
        'tokenizer_present': (SNAPSHOT / 'cache/openpi/big_vision/paligemma_tokenizer.model').is_file(),
        'migration_verified': VERIFICATION.is_file() and json.loads(VERIFICATION.read_text())['passed'],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.preflight:
        return
    if not report['migration_verified']:
        raise RuntimeError('Wait for checkpoint migration and SHA-256 verification to finish')
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    policy = policy_config.create_trained_policy(cfg, CHECKPOINT)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
