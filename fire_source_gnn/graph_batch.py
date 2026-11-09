"""Bounded packing of disjoint graphs without mutating reusable input tensors."""
import torch


def pack_fire_chunks(chunks, chunk_size):
    """Coalesce (node_offset, tensor_tuple) records into bounded edge batches.

    Both raw and prepared layouts keep the receptor index at position three.
    Only those indices change; cached source tensors remain read-only.
    """
    if chunk_size < 1:
        raise ValueError('edge_chunk_size must be positive')
    pieces = None
    size = 0
    for node_offset, tensors in chunks:
        count = len(tensors[0])
        first = 0
        while first < count:
            take = min(chunk_size-size, count-first)
            sliced = list(t[first:first+take] for t in tensors)
            if node_offset:
                sliced[3] = sliced[3] + node_offset
            if pieces is None:
                pieces = [[] for _ in sliced]
            for column, value in zip(pieces, sliced):
                column.append(value)
            size += take
            first += take
            if size == chunk_size:
                yield tuple(column[0] if len(column) == 1 else torch.cat(column, dim=0)
                            for column in pieces)
                pieces, size = None, 0
    if size:
        yield tuple(column[0] if len(column) == 1 else torch.cat(column, dim=0)
                    for column in pieces)
