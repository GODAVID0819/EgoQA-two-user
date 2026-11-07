"""在加载模型前实际验证PyTorch多进程张量传输。"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import tempfile
from multiprocessing.connection import Listener
from pathlib import Path
from queue import Empty


def _receive_tensor(data, acknowledgement):
    tensor = data.get(timeout=30)
    acknowledgement.put(tensor.tolist())


def check_worker_transport():
    # Listener使用与multiprocessing.resource_sharer相同的临时套接字路径。
    with Listener(family="AF_UNIX") as listener:
        address = listener.address
    import torch

    context = multiprocessing.get_context("spawn")
    data = context.Queue()
    acknowledgement = context.Queue()
    child = context.Process(target=_receive_tensor, args=(data, acknowledgement))
    child.start()
    try:
        data.put(torch.arange(16, dtype=torch.int64))
        try:
            received = acknowledgement.get(timeout=45)
        except Empty as exc:
            raise RuntimeError("worker tensor transport timed out") from exc
        if received != list(range(16)):
            raise RuntimeError("worker tensor transport changed the payload")
        child.join(timeout=15)
        if child.exitcode != 0:
            raise RuntimeError(f"worker exited with code {child.exitcode}")
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=10)
        data.close()
        acknowledgement.close()
    return {"status": "passed", "tempdir": tempfile.gettempdir(),
            "unix_socket_path_bytes": len(address.encode()), "tensor_items": len(received)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_worker_transport()
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
