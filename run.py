import os
import argparse

from nsds.common import Params
from nsds.commands.subcommand import ArgumentParserWithDefaults, Subcommand

from modules.face_model import FaceModelWrapper


def main():
    parser = ArgumentParserWithDefaults()
    subparsers = parser.add_subparsers()

    SUBCOMMAND_COLLECTIONS = {
        'index': Index(),
    }

    for name, subcommand in SUBCOMMAND_COLLECTIONS.items():
        subparser = subcommand.add_subparser(name, subparsers)

    args = parser.parse_args()
    args.func(args)


class Index(Subcommand):
    def add_subparser(self, name, subparsers):
        description = 'Indexing image database to .json'
        subparser = subparsers.add_parser(name, description=description)

        subparser.add_argument('param_path', type=str)
        subparser.add_argument('-d', '--data_dir', type=str)
        subparser.add_argument('-o', '--output_path', type=str)
        subparser.add_argument('-m', '--mode', type=str,
                               choices=['single', 'many', 'center'],
                               default='center')

        subparser.set_defaults(func=index_images)
        return subparser


def index_images(args):
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    params = Params.from_file(args.param_path)
    params.pop('vector_search')
    model = FaceModelWrapper(params)
    model.extract_face_embeddings_dataset(args.data_dir, args.output_path,
                                          mode=args.mode)


if __name__ == '__main__':
    main()
