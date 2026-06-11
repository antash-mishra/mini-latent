from __future__ import annotations

from typing import Any

_STYLE = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 2rem auto;
       max-width: 70rem; color: #1a1a2e; }
h1 { border-bottom: 2px solid #444; padding-bottom: 0.3rem; }
h2 { margin-top: 2.2rem; border-bottom: 1px solid #ccc; padding-bottom: 0.2rem; }
table { border-collapse: collapse; margin: 0.8rem 0; }
th, td { border: 1px solid #bbb; padding: 0.35rem 0.7rem; font-size: 0.9rem; }
th { background: #f0f0f5; text-align: left; }
img { max-width: 100%; border: 1px solid #ddd; margin: 0.4rem 0; }
.note { color: #555; font-size: 0.9rem; }
.fail { color: #a02020; }
.ok { color: #207040; }
code { background: #f5f5f7; padding: 0.05rem 0.3rem; font-size: 0.85rem; }
"""


def render_html_report(
    report: dict[str, Any],
    *,
    recon_images: dict[str, str],
    generation_images: dict[str, Any],
    prompt_images: dict[str, str],
) -> str:
    recon = report["reconstruction"]
    generation = report["class_generation"]
    failures = report["failures"]
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>tiny3dlatent generation report</title>",
        f"<style>{_STYLE}</style></head><body>",
        "<h1>tiny3dlatent generation report</h1>",
        f"<p class='note'>created {report['created_at']} &middot; "
        f"run <code>{report['run_dir']}</code></p>",
    ]

    parts.append("<h2>1. Reconstruction (VAE)</h2>")
    parts.append(
        f"<p>checkpoint <code>{recon['checkpoint']}</code> &middot; "
        f"{recon['val_examples']} val examples &middot; "
        f"mean IoU <b>{recon['mean_iou']:.3f}</b> "
        f"(min {recon['min_iou']:.3f}, max {recon['max_iou']:.3f})</p>"
    )
    parts.append("<h3>Best reconstructions</h3>")
    parts.append(f"<img src='images/{recon_images['best_recons']}'>")
    parts.append("<h3>Worst reconstructions</h3>")
    parts.append(f"<img src='images/{recon_images['worst_recons']}'>")

    parts.append("<h2>2. Class-conditional generation</h2>")
    totals = generation["totals"]
    parts.append(
        f"<p>checkpoint <code>{generation['checkpoint']}</code> &middot; "
        f"{totals['samples']} samples: "
        f"non-empty {totals['non_empty']}/{totals['samples']}, "
        f"single-component {totals['single_component']}/{totals['samples']}, "
        f"voxel-count-in-range {totals['voxel_count_in_range']}/{totals['samples']}</p>"
    )
    parts.append(
        _table(
            ["class", "non-empty", "single component", "in voxel range"],
            [
                [
                    name,
                    f"{stats['non_empty']}/{stats['samples']}",
                    f"{stats['single_component']}/{stats['samples']}",
                    f"{stats['voxel_count_in_range']}/{stats['samples']}",
                ]
                for name, stats in generation["per_class"].items()
            ],
        )
    )
    if "mesh_grid" in generation_images:
        parts.append(f"<img src='images/{generation_images['mesh_grid']}'>")
    parts.append("<h3>Turntables (one sample per class)</h3>")
    for strip in generation_images.get("turntables", []):
        parts.append(f"<img src='images/{strip}'>")

    parts.append("<h2>3. Mesh quality</h2>")
    parts.append(
        _table(
            ["class", "vertices", "faces", "watertight", "components", "bounds ok"],
            [
                [
                    row["shape_type"],
                    row["vertices"],
                    row["faces"],
                    _flag(row["watertight"]),
                    row["components"],
                    _flag(row["bounds_ok"]),
                ]
                for row in report["mesh_quality"]
            ],
        )
    )

    parts.append("<h2>4. Prompt generation</h2>")
    prompt_stats = report["prompt_generation"]["prompts"]
    parts.append(
        _table(
            ["prompt", "non-empty", "aspect check (tall/wide)"],
            [
                [
                    prompt,
                    f"{stats['non_empty']}/{stats['samples']}",
                    stats["aspect_check"] or "n/a",
                ]
                for prompt, stats in prompt_stats.items()
            ],
        )
    )
    if "gallery" in prompt_images:
        parts.append(f"<img src='images/{prompt_images['gallery']}'>")

    parts.append("<h2>5. Failure gallery</h2>")
    parts.append(f"<p class='note'>{failures['note']}</p>")
    parts.append(
        _table(
            ["kind", "subject", "note"],
            [
                [item["kind"], item["subject"], item["note"]]
                for item in failures["items"]
            ],
        )
    )

    parts.append("</body></html>")
    return "\n".join(parts)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{header}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def _flag(value: bool) -> str:
    return "<span class='ok'>yes</span>" if value else "<span class='fail'>NO</span>"
