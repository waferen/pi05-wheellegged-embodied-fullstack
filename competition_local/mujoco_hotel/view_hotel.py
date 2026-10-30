"""Interactive hotel entry point: original competition keys."""
from native_viewer import main
if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--view',choices=['Follow','Overview','Head','Hand'],default='Overview')
    main(parser.parse_args().view)
