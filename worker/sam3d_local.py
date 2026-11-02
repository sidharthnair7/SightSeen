"""SAM 3D Objects (Meta) on this PC's 8 GB RTX 3070, for rebuild_things.py's build step when there's no paid GPU. Meta
says the model needs 32 GB; it fits in 8 because only the part that's working sits on the card: every model loads
into ordinary memory (the two big flow transformers and the image encoders in bfloat16, which they compute in
anyway), and each step of the pipeline moves its own models onto the card and back off when it's done: the sparse
structure step (its generator, decoder and image encoder), then the detail step (its generator and encoder), then the
decoders. The depth model never moves: the room's own depth is handed in as the pointmap.

Runs inside WSL, in the conda env from SAM 3D's doc/setup.md (~/sam3d, made by worker/wsl/setup_sam3d.sh):
    python worker/sam3d_local.py <jobs.json>
jobs.json: {"jobs": [{"photo": png, "outlines": [png, ...], "pointmap": npz, "out": folder}], "seed": 42}, paths as
WSL sees them. Each outline's result goes to <out>/<n>.npz (vertices in the photo's OpenCV camera axes, faces,
colours) and <out>/<n>.json (SAM 3D's layout, seconds, the card's peak memory, or the error). The weights come from
Hugging Face the first time (gated: HF_TOKEN in the repo's .env, after Meta approves the request).
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "true")  # as SAM 3D's notebook/inference.py sets it
SAM3D = Path.home() / "sam3d" / "sam-3d-objects"
WEIGHTS = Path.home() / "sam3d" / "weights"
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
BIG = 200_000_000  # parameters: models this size or bigger are kept in bfloat16
sys.path.append(str(SAM3D / "notebook"))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def _token():
    token = os.environ.get("HF_TOKEN")
    if not token and ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == "HF_TOKEN" and value.strip():
                token = value.strip().strip('"')
    return token


def _weights():
    config = WEIGHTS / "hf" / "pipeline.yaml"
    if not config.exists():
        from huggingface_hub import snapshot_download

        token = _token()
        if not token:
            sys.exit("HF_TOKEN isn't set: put it in the repo's .env after Hugging Face approves the access request")
        snapshot_download("facebook/sam-3d-objects", local_dir=str(WEIGHTS / "hf-download"), token=token, max_workers=4)
        os.replace(WEIGHTS / "hf-download" / "checkpoints", WEIGHTS / "hf")  # as doc/setup.md lays it out
    return config


def _load(config_file):
    """SAM 3D's pipeline with every model in ordinary memory, and its steps moving their own models to the card."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    import inference as notebook
    import sam3d_objects.pipeline.inference_pipeline as base

    def into_memory(self, config, ckpt_path, state_dict_fn=None, state_dict_key="state_dict", device="cuda"):
        # SAM 3D's own loader, except that the model is made and filled in ordinary memory (it builds straight onto
        # the card, and reads safetensors onto it too, which holds two copies of the biggest model at once)
        with torch.device("cpu"):
            model = instantiate(config)
        if ckpt_path.endswith(".safetensors"):
            state_dict = load_file(ckpt_path, device="cpu")
            if state_dict_fn is not None:
                state_dict = state_dict_fn(state_dict)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
        else:
            model = base.load_model_from_checkpoint(model, ckpt_path, strict=True, device="cpu", freeze=True, eval=True,
                                                    state_dict_key=state_dict_key, state_dict_fn=state_dict_fn)
        if sum(p.numel() for p in model.parameters()) >= BIG:
            model = model.to(torch.bfloat16)
        return model

    base.InferencePipeline.instantiate_and_load_from_pretrained = into_memory
    config = OmegaConf.load(config_file)
    config.rendering_engine = "pytorch3d"  # as notebook/inference.py sets it: no nvdiffrast
    config.compile_model = False
    config.workspace_dir = os.path.dirname(config_file)
    notebook.check_hydra_safety(config, notebook.WHITELIST_FILTERS, notebook.BLACKLIST_FILTERS)
    pipeline = instantiate(config)
    if isinstance(getattr(pipeline, "depth_model", None), torch.nn.Module):
        pipeline.depth_model.to("cpu")  # the room's depth replaces it
    torch.cuda.empty_cache()

    models, embedders = pipeline.models, pipeline.condition_embedders
    steps = {
        "sample_sparse_structure": lambda: [models["ss_generator"], models["ss_decoder"],
                                            embedders.get("ss_condition_embedder")],
        "sample_slat": lambda: [models["slat_generator"], embedders.get("slat_condition_embedder")],
        "decode_slat": lambda: [models["slat_decoder_mesh"], models["slat_decoder_gs"],
                                models["slat_decoder_gs_4"] if "slat_decoder_gs_4" in models else None],
    }
    for name, used in steps.items():
        method = getattr(pipeline, name)

        def on_card(*args, _method=method, _used=used, **kwargs):
            parts = [part for part in _used() if isinstance(part, torch.nn.Module)]
            for part in parts:
                part.to("cuda")
            try:
                return _method(*args, **kwargs)
            finally:
                for part in parts:
                    part.to("cpu")
                torch.cuda.empty_cache()

        setattr(pipeline, name, on_card)
    return pipeline


def _run(pipeline, photo, outline, pointmap, seed):
    """One thing: SAM 3D's object moved by its own layout into the photo's camera, in OpenCV's axes."""
    from sam3d_objects.utils.visualization import SceneVisualizer

    rgba = np.concatenate([photo[..., :3], (outline.astype(np.uint8) * 255)[..., None]], axis=-1)
    output = pipeline.run(rgba, None, seed, stage1_only=False, with_mesh_postprocess=False, with_texture_baking=False,
                          with_layout_postprocess=False, use_vertex_color=True, stage1_inference_steps=None,
                          pointmap=pointmap)
    mesh = output["glb"]
    y_up = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)  # to_glb's z-up to y-up turn
    local = np.asarray(mesh.vertices, dtype=np.float64) @ y_up.T
    device = output["rotation"].device
    cloud = SceneVisualizer.object_pointcloud(
        points_local=torch.tensor(local, dtype=torch.float32, device=device).unsqueeze(0),
        quat_l2c=output["rotation"], trans_l2c=output["translation"], scale_l2c=output["scale"])
    in_camera = cloud.points_list()[0].detach().cpu().numpy() * np.array([-1.0, -1.0, 1.0])  # PyTorch3D to OpenCV
    layout = {k: output[k].detach().float().cpu().numpy().tolist() for k in ("rotation", "translation", "scale")}
    return in_camera.astype(np.float32), np.asarray(mesh.faces, np.int32), \
        np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.uint8), layout


def main():
    from PIL import Image

    plan = json.loads(Path(sys.argv[1]).read_text())
    seed = plan.get("seed", 42)
    started = time.perf_counter()
    pipeline = _load(str(_weights()))
    print(f"loaded in {time.perf_counter() - started:.0f} s", flush=True)
    for job in plan["jobs"]:
        out = Path(job["out"])
        out.mkdir(parents=True, exist_ok=True)
        photo = np.array(Image.open(job["photo"]).convert("RGB"))
        pointmap = None
        if job.get("pointmap"):
            pointmap = torch.from_numpy(np.load(job["pointmap"])["pointmap"].astype(np.float32))
        for n, outline_file in enumerate(job["outlines"]):
            if (out / f"{n}.json").exists():
                continue  # done in an earlier run
            outline = np.array(Image.open(outline_file).convert("L")) > 127
            torch.cuda.reset_peak_memory_stats()
            began = time.perf_counter()
            try:
                vertices, faces, colours, layout = _run(pipeline, photo, outline, pointmap, seed)
                np.savez(out / f"{n}.npz", vertices=vertices, faces=faces, colours=colours)
                result = {"vertices": len(vertices), "layout": layout}
            except Exception as error:
                result = {"error": f"{type(error).__name__}: {error}"[:800]}
            result["seconds"] = round(time.perf_counter() - began, 1)
            result["peakGb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            (out / f"{n}.json").write_text(json.dumps(result, indent=2))
            print(job["out"], n, json.dumps({k: v for k, v in result.items() if k != "layout"}), flush=True)


if __name__ == "__main__":
    main()
