# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: annoy.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='annoy.proto',
  package='annoy',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x0b\x61nnoy.proto\x12\x05\x61nnoy\"\x17\n\x06Vector\x12\r\n\x05\x65lems\x18\x01 \x03(\x01\"9\n\rSearchRequest\x12\x1d\n\x06vector\x18\x01 \x01(\x0b\x32\r.annoy.Vector\x12\t\n\x01k\x18\x02 \x01(\x05\"-\n\x0eSearchResponse\x12\x0b\n\x03ids\x18\x01 \x03(\t\x12\x0e\n\x06scores\x18\x02 \x03(\x01\x32@\n\x05\x41nnoy\x12\x37\n\x06Search\x12\x14.annoy.SearchRequest\x1a\x15.annoy.SearchResponse\"\x00\x62\x06proto3')
)




_VECTOR = _descriptor.Descriptor(
  name='Vector',
  full_name='annoy.Vector',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='elems', full_name='annoy.Vector.elems', index=0,
      number=1, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=22,
  serialized_end=45,
)


_SEARCHREQUEST = _descriptor.Descriptor(
  name='SearchRequest',
  full_name='annoy.SearchRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='vector', full_name='annoy.SearchRequest.vector', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='k', full_name='annoy.SearchRequest.k', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=47,
  serialized_end=104,
)


_SEARCHRESPONSE = _descriptor.Descriptor(
  name='SearchResponse',
  full_name='annoy.SearchResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ids', full_name='annoy.SearchResponse.ids', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scores', full_name='annoy.SearchResponse.scores', index=1,
      number=2, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=106,
  serialized_end=151,
)

_SEARCHREQUEST.fields_by_name['vector'].message_type = _VECTOR
DESCRIPTOR.message_types_by_name['Vector'] = _VECTOR
DESCRIPTOR.message_types_by_name['SearchRequest'] = _SEARCHREQUEST
DESCRIPTOR.message_types_by_name['SearchResponse'] = _SEARCHRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Vector = _reflection.GeneratedProtocolMessageType('Vector', (_message.Message,), {
  'DESCRIPTOR' : _VECTOR,
  '__module__' : 'annoy_pb2'
  # @@protoc_insertion_point(class_scope:annoy.Vector)
  })
_sym_db.RegisterMessage(Vector)

SearchRequest = _reflection.GeneratedProtocolMessageType('SearchRequest', (_message.Message,), {
  'DESCRIPTOR' : _SEARCHREQUEST,
  '__module__' : 'annoy_pb2'
  # @@protoc_insertion_point(class_scope:annoy.SearchRequest)
  })
_sym_db.RegisterMessage(SearchRequest)

SearchResponse = _reflection.GeneratedProtocolMessageType('SearchResponse', (_message.Message,), {
  'DESCRIPTOR' : _SEARCHRESPONSE,
  '__module__' : 'annoy_pb2'
  # @@protoc_insertion_point(class_scope:annoy.SearchResponse)
  })
_sym_db.RegisterMessage(SearchResponse)



_ANNOY = _descriptor.ServiceDescriptor(
  name='Annoy',
  full_name='annoy.Annoy',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=153,
  serialized_end=217,
  methods=[
  _descriptor.MethodDescriptor(
    name='Search',
    full_name='annoy.Annoy.Search',
    index=0,
    containing_service=None,
    input_type=_SEARCHREQUEST,
    output_type=_SEARCHRESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_ANNOY)

DESCRIPTOR.services_by_name['Annoy'] = _ANNOY

# @@protoc_insertion_point(module_scope)