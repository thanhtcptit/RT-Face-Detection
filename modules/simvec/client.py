import os
import grpc

from modules.simvec.rpc import annoy_pb2
from modules.simvec.rpc import annoy_pb2_grpc


def generate_repeated_var(repeated_var):
    for elem in repeated_var:
        yield elem


class AnnoyClient:
    def __init__(self, port):
        self._port = port

    def search(self, emb_vector, k):
        try:
            with grpc.insecure_channel(f'localhost:{self._port}') as channel:
                stub = annoy_pb2_grpc.AnnoyStub(channel)
                response = stub.Search(annoy_pb2.SearchRequest(
                    vector=annoy_pb2.Vector(
                        elems=generate_repeated_var(emb_vector)),
                    k=k))
                return response.ids, response.scores
        except Exception as e:
            return [], []


if __name__ == '__main__':
    import numpy as np
    client = AnnoyClient(50051)
    ids, _ = client.search(np.random.rand(128), 10)
    print(len(ids))
