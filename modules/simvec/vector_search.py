import json

from tqdm import tqdm
from multiprocessing import Manager
from annoy import AnnoyIndex as VectorIndex


class VectorSearch:
    def __init__(self, data_file, dims, metric='angular'):
        self.data_file = data_file
        self.dims = dims

        self.index = VectorIndex(dims, metric=metric)

        self.key_dict = Manager().dict()
        self.id_dict = Manager().dict()

        self.build()

    def build(self):
        self._index_from_file()
        self.index.build(-1)
        print('Finish build tree')

    def _index_from_file(self):
        def __index_vector(key, v):
            i = self.id_dict.get(key)
            if i is None:
                i = self.index.get_n_items()
                self.key_dict[i] = key
                self.id_dict[key] = i
            self.index.add_item(i, v)

        with open(self.data_file, 'r') as f_r:
            for line in tqdm(f_r):
                r = json.loads(line)
                key = r['key']
                v = r['embedding']
                __index_vector(key, v)

    def search(self, vector, k=50):
        status = 0
        ids = []
        scores = []

        if len(vector) != self.dims:
            raise ValueError(
                f'Wrong vector dimension: {len(vector)} != {self.dims}')

        I = self.index.get_nns_by_vector(vector, k, -1, True)

        ids = list(filter(lambda i: i is not None,
                   [self.key_dict.get(i) for i in I[0]]))
        scores = I[1]
        return ids, scores
