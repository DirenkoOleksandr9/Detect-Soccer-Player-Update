# Run in GitHub Codespaces

1. Open the repo in Codespaces (Use Template > Create Codespace on main).
2. Devcontainer builds automatically; deps install on first start.
3. Place your video into the workspace root or upload via Codespaces UI.
4. Run detection only (fastest on CPU):

```bash
python3 main_pipeline.py 9.mp4 --stage detection --output-dir output
```

5. Full pipeline (CPU-only is slow; for short clips):

```bash
python3 main_pipeline.py 9.mp4 --player-id 1 --output-dir output
```

6. Outputs appear under `output/`. Download via Codespaces file explorer or `Download Artifact` if using Actions workflow.

Note: Codespaces does not provide GPUs. For GPU runs, use the Colab notebook.
