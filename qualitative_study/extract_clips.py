#!/usr/bin/env python3
"""
Extract the observed clip for each qualitative sample and build a merged data.json
for the annotation UI.

Pipeline (stream & cut):
  * Read baseline_preds.json and modified_preds.json (paired by position; both files
    were produced with the same --seed so sample i is the same clip in both).
  * For each sample, ffmpeg seeks to the observed window in the OFFICIAL EPIC-Kitchens
    video (over HTTPS, using range requests) and writes only that ~30s window to
    clips/clip_XXX.mp4 -- no full video is ever stored on disk.
  * Write data.json: one entry per successfully extracted clip with the narration,
    ground-truth action, and both models' top-5 predictions.

Robustness: a missing video URL, an ffmpeg failure, or a timeout is logged and the
clip is skipped -- the run never crashes. Already-extracted clips are reused, so the
script is resumable.

Usage (from the qualitative_study/ folder):
  python extract_clips.py
  python extract_clips.py --baseline ../baseline_preds.json --modified ../modified_preds.json
  python extract_clips.py --limit 5          # quick smoke test on the first 5 clips
  python extract_clips.py --reencode copy     # stream-copy instead of re-encoding
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def log(msg):
    print(msg, flush=True)


def find_ffmpeg(explicit=None):
    """Locate an ffmpeg binary: explicit path, then PATH, then the imageio-ffmpeg bundle."""
    if explicit:
        return explicit
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def parse_timestamp(ts):
    """'HH:MM:SS.cc' (or a bare number of seconds) -> float seconds."""
    if ts is None:
        return None
    ts = str(ts).strip()
    if ":" in ts:
        h, m, s = ts.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    return float(ts)


def ground_truth_action(sample):
    """Prefer the canonical action label; fall back to the narration."""
    target = sample.get("target", {}) or {}
    actions = target.get("actions") or []
    if actions and actions[0].get("name"):
        return actions[0]["name"]
    return target.get("narration")


def top_predictions(sample, k):
    """Return [{label, score}] for the sample's top-k future actions."""
    preds = (sample.get("predictions", {}) or {}).get("top_actions", []) or []
    out = []
    for p in preds[:k]:
        out.append({"label": p.get("action"), "score": round(float(p.get("probability", 0.0)), 6)})
    return out


def build_ffmpeg_cmd(ffmpeg, url, start, duration, out_path, mode):
    """ffmpeg command that seeks to `start` and writes `duration` seconds of `url`.

    -ss before -i = input seeking: over HTTPS this uses range requests to jump to the
    window instead of downloading the whole file. -reconnect* makes the HTTP read
    resilient to dropped connections.
    """
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
        "-ss", f"{start:.3f}", "-i", url, "-t", f"{duration:.3f}",
    ]
    if mode == "copy":
        # Fast, no decode; clip starts at the nearest keyframe <= start. -avoid_negative_ts
        # keeps timestamps sane for browser playback. Map only video+audio, drop data tracks.
        cmd += ["-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", "-avoid_negative_ts", "make_zero"]
    else:
        # Re-encode to a guaranteed browser-friendly H.264/AAC mp4 with the moov atom up
        # front. Drops the timecode data track that EPIC videos carry.
        cmd += [
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
        ]
    cmd += ["-movflags", "+faststart", str(out_path)]
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default=None, help="baseline preds JSON (default: next to this script, else repo root)")
    ap.add_argument("--modified", default=None, help="modified preds JSON (default: next to this script, else repo root)")
    ap.add_argument("--urls", default=str(HERE / "video_urls.json"), help="video_id -> source URL map")
    ap.add_argument("--clips-dir", default=str(HERE / "clips"))
    ap.add_argument("--out", default=str(HERE / "data.json"))
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--reencode", choices=["encode", "copy"], default="encode",
                    help="'encode' (robust, browser-safe) or 'copy' (faster, keyframe-aligned).")
    ap.add_argument("--ffmpeg", default=None, help="Path to ffmpeg (defaults to PATH / imageio-ffmpeg).")
    ap.add_argument("--timeout", type=int, default=300, help="Per-clip ffmpeg timeout, seconds.")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N samples.")
    ap.add_argument("--overwrite", action="store_true", help="Re-extract clips that already exist.")
    args = ap.parse_args()

    ffmpeg = find_ffmpeg(args.ffmpeg)
    if not ffmpeg:
        log("ERROR: ffmpeg not found. Install with `python -m pip install imageio-ffmpeg` or add ffmpeg to PATH.")
        sys.exit(1)
    log(f"Using ffmpeg: {ffmpeg}")

    # The preds JSONs may live next to this script or one level up (repo root).
    def resolve_preds(arg, name):
        if arg:
            return arg
        for cand in (HERE / name, HERE.parent / name):
            if cand.exists():
                return str(cand)
        return str(HERE / name)  # fall through; open() will raise a clear error

    args.baseline = resolve_preds(args.baseline, "baseline_preds.json")
    args.modified = resolve_preds(args.modified, "modified_preds.json")
    log(f"baseline preds: {args.baseline}")
    log(f"modified preds: {args.modified}")

    baseline = json.load(open(args.baseline, encoding="utf-8"))
    modified = json.load(open(args.modified, encoding="utf-8"))
    urls = json.load(open(args.urls, encoding="utf-8"))

    if len(baseline) != len(modified):
        log(f"WARNING: baseline ({len(baseline)}) and modified ({len(modified)}) differ in length; "
            f"pairing the first {min(len(baseline), len(modified))} by position.")
    n = min(len(baseline), len(modified))
    if args.limit:
        n = min(n, args.limit)

    clips_dir = Path(args.clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)

    data, skipped = [], []
    for i in range(n):
        b, m = baseline[i], modified[i]
        clip_id = f"clip_{i + 1:03d}"
        vid = b.get("sample", {}).get("video_id")

        # Sanity: the two files should describe the same underlying sample at this position.
        if vid != m.get("sample", {}).get("video_id") or \
           b.get("sample", {}).get("index") != m.get("sample", {}).get("index"):
            log(f"[{clip_id}] SKIP: baseline/modified describe different samples at this position.")
            skipped.append((clip_id, "sample mismatch"))
            continue

        obs = b.get("observed", {}) or {}
        start = parse_timestamp(obs.get("start"))
        duration = obs.get("duration_sec")
        if duration is None and obs.get("end") is not None and start is not None:
            duration = parse_timestamp(obs.get("end")) - start
        if start is None or duration is None or duration <= 0:
            log(f"[{clip_id}] SKIP {vid}: bad observed window (start={obs.get('start')}, dur={duration}).")
            skipped.append((clip_id, "bad window"))
            continue

        url = urls.get(vid)
        out_path = clips_dir / f"{clip_id}.mp4"
        rel_path = f"clips/{clip_id}.mp4"

        if url is None:
            log(f"[{clip_id}] SKIP: no source URL for video_id {vid}.")
            skipped.append((clip_id, f"no url for {vid}"))
            continue

        if out_path.exists() and out_path.stat().st_size > 0 and not args.overwrite:
            log(f"[{clip_id}] reuse existing {rel_path}")
        else:
            cmd = build_ffmpeg_cmd(ffmpeg, url, start, duration, out_path, args.reencode)
            log(f"[{clip_id}] {vid}  {obs.get('start')} +{duration:.2f}s  -> {rel_path}")
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                log(f"[{clip_id}] SKIP {vid}: ffmpeg timed out after {args.timeout}s.")
                out_path.unlink(missing_ok=True)
                skipped.append((clip_id, "timeout"))
                continue
            except Exception as e:
                log(f"[{clip_id}] SKIP {vid}: ffmpeg failed to launch ({e!r}).")
                skipped.append((clip_id, "launch error"))
                continue
            if r.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
                err = (r.stderr or "").strip().splitlines()
                log(f"[{clip_id}] SKIP {vid}: ffmpeg rc={r.returncode}. {err[-1] if err else ''}")
                out_path.unlink(missing_ok=True)
                skipped.append((clip_id, f"ffmpeg rc={r.returncode}"))
                continue

        data.append({
            "clip_id": clip_id,
            "video_file": rel_path,
            "video_id": vid,
            "observed_window": f"{obs.get('start')} → {obs.get('end')}",
            "narration": (b.get("target", {}) or {}).get("narration"),
            "ground_truth_action": ground_truth_action(b),
            "baseline": top_predictions(b, args.topk),
            "modified": top_predictions(m, args.topk),
        })

    json.dump(data, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    log("")
    log(f"Done. {len(data)} clips written to {args.out}; {len(skipped)} skipped.")
    for cid, reason in skipped:
        log(f"  skipped {cid}: {reason}")


if __name__ == "__main__":
    main()
