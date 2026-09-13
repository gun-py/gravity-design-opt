# Running this project on Colab from the Mac terminal

How I run code for this competition: a GPU VM rented through the `colab` CLI, with the
dataset read from Google Drive. The CLI lives at `/opt/miniconda3/bin/colab`.

## Mental model

- A **session** is a live Jupyter kernel on a rented Colab VM. `colab new` allocates it
  (billable), `colab stop` releases it. Nothing else reclaims it except a ~24h cap, so
  **always stop when done**.
- **Kernel state persists** across `colab exec` calls in the same session (imports,
  variables, functions survive). Build up state incrementally.
- The VM is **ephemeral**: an idle/terminated VM loses `/content` and the Drive mount.
  Persist anything important to Drive and/or download it locally.

## 1. Auth (once)

Check status; the token auto-refreshes:

```bash
colab whoami        # email, scopes, expiry
colab sessions      # lists active server sessions
```

Needs the `colaboratory` and `drive.file` scopes (already set up here). If a call 403s,
re-mint ADC per `colab skill`.

## 2. Create a GPU session

```bash
colab new -s biohub --gpu T4
colab status -s biohub          # confirm Hardware: T4, Status: IDLE
```

Supported `--gpu`: `T4, L4, G4, H100, A100` (availability is tier-gated; T4 is the
reliable default). Go higher only if a job needs it. Omit `--gpu` for CPU.

## 3. Mount Google Drive (interactive — run in YOUR terminal)

The data lives at `/content/drive/MyDrive/Data/kaggle/biohub`. Drive mounting is a
browser OAuth flow with a **~120s window**, so run it yourself in a real terminal — it
cannot be automated from the agent:

```bash
colab drivemount -s biohub
```

1. It prints a URL — open it immediately, authorize as your Google account.
2. Return to the terminal and press **Enter**.
3. It prints `Mounted at /content/drive`.

The mount then persists on the VM for all later `colab exec` calls. It survives
`colab restart-kernel` but **not** VM loss — remount after any `colab new`.

## 4. Run code on the VM

Preferred: write a local `.py` and run it on the remote kernel.

```bash
colab exec -s biohub -f script.py          # runs local file on the VM
echo "print(1)" | colab exec -s biohub     # pipe code
colab upload -s biohub local.py /content/local.py   # for import reuse
colab install -s biohub zarr geff          # pip install on the VM
```

**The exec client times out at ~120s** ("Timeout waiting for output") while the kernel
keeps running. For long jobs, don't rely on stdout — have the script **write results to
Drive**, let the client time out, wait, then read the file back:

```python
# inside the long job
work = "/content/drive/MyDrive/Data/kaggle/biohub_work"
open(work + "/result.txt", "w").write(summary)
```

```bash
# after waiting long enough, kernel now idle:
echo "print(open('/content/drive/MyDrive/Data/kaggle/biohub_work/result.txt').read())" \
  | colab exec -s biohub
```

Check whether the kernel is free with `colab status -s biohub` (a metadata call, does
not queue on the kernel).

## 5. Use the GPU (cupy)

The T4 is used through `cupy` / `cupyx` (preinstalled on Colab). Pattern with a CPU
fallback:

```python
try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter, maximum_filter
    GPU = True
except Exception:
    from scipy.ndimage import gaussian_filter, maximum_filter
    GPU = False

v = cp.asarray(volume, cp.float32)      # host -> device
# ... filter on device ...
result = cp.asnumpy(v)                   # device -> host
```

Verify the GPU: `nvidia-smi` via exec, or `cp.cuda.runtime.getDeviceProperties(0)`.

## 6. Read data from Drive (it is slow per-file)

Drive FUSE has high per-file latency, so:

- **Never `os.walk` a `.zarr`** — chunk trees have millions of files and wedge the
  kernel. List one level with `os.listdir`.
- Read many small files (e.g. `.geff` graphs) concurrently:

```python
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=32) as ex:
    results = list(ex.map(read_one, files))
```

- For repeated access to a big image, copy it to local VM disk once, then read locally:

```python
import shutil, os
dst = "/content/vid.zarr"
if not os.path.exists(dst):
    shutil.copytree(src_on_drive, dst)   # ~1 min per 800 MB video
```

Images are OME-Zarr v3, opened as `zarr.open(path)["0"]`, shape `(T, Z, Y, X)`.

## 7. Save and fetch outputs

Write to `.../biohub/biohub_work/` on Drive (persists across VMs), and pull copies down:

```bash
colab download -s biohub /content/drive/MyDrive/Data/kaggle/biohub_work/submission.csv work/submission.csv
```

## 8. Stop the session (always)

```bash
colab stop -s biohub
colab sessions            # confirm none active
```

## Full resume-from-scratch sequence

```bash
colab new -s biohub --gpu T4
colab drivemount -s biohub          # in your terminal: authorize, press Enter
colab install -s biohub zarr geff   # fresh VM has no extra libs
# ... colab exec -s biohub -f <script>.py ... (save results to biohub_work/)
colab stop -s biohub
```

## Gotchas seen in practice

- Sessions get released after idle (~90 min) or if the keep-alive daemon dies —
  losing the mount and `/content`. Recreate when `colab sessions` shows none.
- `colab exec -f nb.ipynb` runs a notebook, but any single cell over ~120s trips the
  client timeout; keep long-running work in scripts that persist to Drive.
- `colab repl`, `colab console`, `colab auth`, `colab drivemount` expect a TTY — run
  them yourself, not from the agent.
