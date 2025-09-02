import torch


def _inject_beacon_in_doc(
    doc: torch.Tensor, beacon_id: int, stride: int
) -> torch.Tensor:
    """Insert a beacon after every `stride` tokens. No padding; preserves all tokens."""
    assert doc.ndim == 1
    if stride <= 0:
        return doc
    L = doc.numel()
    n_full = L // stride
    if n_full == 0:
        return doc

    chunks = doc[: n_full * stride].view(n_full, stride)  # [n_full, stride]
    beacons = torch.full(
        (n_full, 1), beacon_id, dtype=doc.dtype, device=doc.device
    )  # [n_full, 1]
    interleaved = torch.cat([chunks, beacons], dim=1).reshape(-1)  # [(stride+1)*n_full]
    remainder = doc[n_full * stride :]  # [L - n_full*stride]
    return torch.cat([interleaved, remainder], dim=0)


def inject_beacon_to_docs(
    input_ids: torch.Tensor, bos_id: int, beacon_id: int, stride: int
) -> torch.Tensor:
    """Inject beacon tokens into multiple documents separated by BOS tokens."""
    assert input_ids.ndim == 1, "input_ids must be a 1D tensor"

    bos_idxs = torch.nonzero(input_ids == bos_id, as_tuple=False).squeeze(1)

    if len(bos_idxs) <= 1:
        return _inject_beacon_in_doc(input_ids, beacon_id, stride)

    # Process each document segment
    processed_docs = []
    for i in range(len(bos_idxs) - 1):
        start, end = bos_idxs[i], bos_idxs[i + 1]
        doc = input_ids[start:end]
        processed_docs.append(_inject_beacon_in_doc(doc, beacon_id, stride))
    processed_docs.append(
        _inject_beacon_in_doc(input_ids[bos_idxs[-1] :], beacon_id, stride)
    )

    return torch.cat(processed_docs)


if __name__ == "__main__":
    input_ids = torch.tensor(
        [0, 2, 4, 6, 8, 0, 1, 3, 5, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    )
    beacon_id = 69
    bos_id = 0
    stride = 4
    print(inject_beacon_to_docs(input_ids, bos_id, beacon_id, stride))
