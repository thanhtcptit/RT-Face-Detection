# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import modules.simvec.rpc.annoy_pb2 as annoy__pb2


class AnnoyStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Search = channel.unary_unary(
        '/annoy.Annoy/Search',
        request_serializer=annoy__pb2.SearchRequest.SerializeToString,
        response_deserializer=annoy__pb2.SearchResponse.FromString,
        )


class AnnoyServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def Search(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_AnnoyServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Search': grpc.unary_unary_rpc_method_handler(
          servicer.Search,
          request_deserializer=annoy__pb2.SearchRequest.FromString,
          response_serializer=annoy__pb2.SearchResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'annoy.Annoy', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))