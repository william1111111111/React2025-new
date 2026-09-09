# HiRP scratch workspace

Independent HiRP development moved from ../react2025. Source, tests and the
initial Git history live here. Original experiments and legacy networks remain
in ../react2025.

Run CPU tests from this directory:

```bash
.venv/bin/python -m pytest hirp/tests -v
```

Local symlinks (not committed):
- data -> /home/zhengshiyi/react/data
- .venv -> /home/zhengshiyi/react2025/.venv
- external/FaceVerse/mean_face.npy and std_face.npy -> original normalization files

Data and the Python environment are shared, not copied; their original paths
must remain available. No old model or checkpoint is linked into this workspace.
See hirp/README.md for architecture, API and validation details.
