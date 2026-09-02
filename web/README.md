# QField-Dyn web portal

This portal exposes the frozen QField-Dyn inference path through a small upload–predict–download interface.

## Supported ZIP

The ZIP root contains one selected tier:

```text
T1/
  ids.txt
  example-1/
    example-1.pdb
    example-1_obs.xtc
    meta.json
```

`meta.json` must follow the official T1–T4 temporal specifications. Only topology, metadata and observed frames are accepted. The output ZIP contains `{tier}/{id}_pred.xtc` files.

## Run

Place `web/` in the QField-Dyn reproducibility package root, then run with the package environment:

```bash
QFIELD_WEB_HOST=127.0.0.1 QFIELD_WEB_PORT=8765 .venv/bin/python web/server.py
```

Open `http://127.0.0.1:8765`.

The inference worker is deliberately single-GPU and processes one job at a time. Uploads are limited to 512 MiB.
