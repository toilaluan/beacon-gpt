import torch


def _inject_beacon_in_doc(
    doc: torch.Tensor, beacon_id: int, stride: int
) -> torch.Tensor:
    """Inject beacon tokens into a document at regular stride intervals."""
    # Pad to multiple of stride
    n_pad = stride - doc.size(0) % stride
    doc = torch.nn.functional.pad(doc, (0, n_pad), value=0)
    doc_length = doc.size(0)
    n_beacon = doc_length // stride

    if n_beacon == 0:
        return doc

    # Create beacon tensor once
    beacon_tensor = torch.full(
        (n_beacon,), beacon_id, dtype=doc.dtype, device=doc.device
    )

    # Reshape doc into chunks of stride length
    doc_chunks = doc[: n_beacon * stride].view(n_beacon, stride)

    # Interleave doc chunks with beacon tokens
    result = torch.cat([doc_chunks, beacon_tensor.unsqueeze(1)], dim=1)
    result = result.view(-1)

    # Append remaining tokens if any
    remainder = doc[n_beacon * stride :]
    if remainder.numel() > 0:
        result = torch.cat([result, remainder])
    return result[: -(n_pad + 1)]


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
