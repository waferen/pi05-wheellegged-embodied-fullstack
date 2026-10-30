"""Visible MuJoCo demo using the migrated 40000-step policy."""
import argparse
from native_viewer import main
from pi05_teleop import Pi05Teleop

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--execute-steps', type=int, default=25)
    parser.add_argument('--view', choices=['Follow', 'Overview', 'Head', 'Hand'], default='Follow')
    args = parser.parse_args()
    main(args.view, sim_factory=lambda: Pi05Teleop(args.host, args.port, args.execute_steps))
