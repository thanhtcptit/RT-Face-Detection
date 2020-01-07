import os
import sys
import time
import grpc
import argparse

from concurrent import futures

from modules.simvec.rpc import annoy_pb2
from modules.simvec.rpc import annoy_pb2_grpc
from modules.simvec.vector_search import VectorSearch


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=50051)
    parser.add_argument('-f', '--data_file', type=str)
    parser.add_argument('-d', '--dims', type=int)
    return parser.parse_args()


def generate_repeated_var(repeated_var):
    for elem in repeated_var:
        yield elem


class AnnoyServicer(annoy_pb2_grpc.AnnoyServicer):
    def __init__(self, data_file, dims):
        self._vector_search = VectorSearch(data_file, dims)

    def Search(self, request, context):
        emb_vector, k = request.vector.elems, request.k
        ids, scores = self._vector_search.search(emb_vector, k)
        return annoy_pb2.SearchResponse(
            ids=generate_repeated_var(ids),
            scores=generate_repeated_var(scores))


def start(server, port, data_file, dims):
    annoy_pb2_grpc.add_AnnoyServicer_to_server(
        AnnoyServicer(data_file, dims), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print('Service start')


def serve(port, data_file, dims):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    start(server, port, data_file, dims)
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    args = parse_args()
    serve(args.port, args.data_file, args.dims)
