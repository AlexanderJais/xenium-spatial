"""
pages/2_consensus_panel.py
Consensus Panel — the gene set the pipeline actually runs on.

Different sections can carry slightly different add-on (custom) gene panels.
Comparing samples on a gene that is only on *some* panels is unsafe: a gene
absent from a panel reads as "not expressed" and can masquerade as differential
expression. To avoid that, the pipeline runs on the **consensus panel** — the
strict intersection of every configured sample's panel (base genes in all +
add-on genes in all). Nothing is ever zero-filled, so every consensus gene is
genuinely measured in every sample.

This page computes the consensus from each slide's ``features.tsv.gz`` metadata
(no matrix load), shows what is kept and what is excluded, and locks it in as
the active panel for Sample PCA, clustering and all downstream quantification.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

import sys as _sys; _sys.path.insert(0, str(Path(__file__).parent.parent))
from ui_utils import inject_css, page_header, init_session_state

st.set_page_config(page_title="Consensus Panel · Xenium Spatial Pipeline", page_icon="🧬",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()
init_session_state()

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from xenium_spatial.panel_registry import PanelRegistry  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────
def _valid_dir(p: str) -> bool:
    return bool(p) and (Path(p) / "cell_feature_matrix" / "matrix.mtx.gz").exists()


@st.cache_data(show_spinner=False)
def _gene_set(run_dir: str) -> tuple:
    """Gene-Expression gene names for a slide, read from features.tsv.gz.

    Returns ``(frozenset_of_genes, error_str)``; cached on ``run_dir`` since the
    metadata is static during a session.
    """
    feat = Path(run_dir) / "cell_feature_matrix" / "features.tsv.gz"
    if not feat.exists():
        return frozenset(), f"features.tsv.gz not found in {run_dir}"
    try:
        df = pd.read_csv(feat, sep="\t", header=None,
                         names=["gene_id", "gene_name", "feature_type"], compression="gzip")
        genes = df.loc[df["feature_type"] == "Gene Expression", "gene_name"].astype(str)
        return frozenset(genes), ""
    except Exception as e:  # noqa: BLE001
        return frozenset(), f"Error reading {feat.name}: {e}"


def _chips(genes, colour="#0F2E52", bg="#EAF0F7", limit=None):
    shown = genes if limit is None else genes[:limit]
    html = " ".join(
        f'<span style="display:inline-block;background:{bg};color:{colour};'
        f'border-radius:4px;padding:1px 7px;margin:2px;font-size:12px;'
        f'font-family:monospace;">{g}</span>' for g in shown
    )
    if limit is not None and len(genes) > limit:
        html += f' <span style="color:#8A95A3;font-size:12px;">+{len(genes) - limit} more…</span>'
    return html


# ── Page ──────────────────────────────────────────────────────────────────────
page_header("🧬 Consensus Panel",
            "The genes measured in every sample — what the whole pipeline runs on")

slides = st.session_state.get("slides", [])
valid = [s for s in slides if _valid_dir(s.get("run_dir", ""))]
if len(valid) < 2:
    st.warning("Need at least 2 valid slides to build a consensus. Configure them in "
               "**📁 Study Setup**.")
    st.stop()

# Read each slide's panel + flag any that couldn't be read.
gene_sets, read_errors = {}, []
for s in valid:
    gs, err = _gene_set(s["run_dir"])
    if err:
        read_errors.append((s["slide_id"], err))
    else:
        gene_sets[s["slide_id"]] = set(gs)

if read_errors:
    for sid, err in read_errors:
        st.error(f"**{sid}**: {err}")
if len(gene_sets) < 2:
    st.stop()

try:
    registry = PanelRegistry(st.session_state["base_panel_csv"])
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load the base panel CSV: {e}")
    st.stop()

con = registry.consensus_panel(gene_sets)
n_base, n_addon = len(con["base"]), len(con["addon"])
n_total = len(con["consensus"])
n_samples = len(con["slides"])

# ── Headline ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Consensus genes", f"{n_total:,}")
c2.metric("Base panel", f"{n_base:,}")
c3.metric("Shared add-on", f"{n_addon:,}")
c4.metric("Samples", f"{n_samples}")

st.markdown(
    f"The pipeline runs on **{n_total} consensus genes** "
    f"= **{n_base} base** + **{n_addon} shared add-on**, "
    f"each present in **all {n_samples}** samples (strict intersection — no zero-filling)."
)

# Base-panel coverage (normally complete; flag any gene missing from a sample).
if con["excluded_base"]:
    st.warning(
        f"⚠️ **{len(con['excluded_base'])} base-panel gene(s)** are missing from at least one "
        "sample and were **excluded** from the consensus (strict intersection): "
        + ", ".join(con["excluded_base"][:20])
        + (" …" if len(con["excluded_base"]) > 20 else "")
    )
else:
    st.success(f"✅ All {n_base} base-panel genes are present in every sample.")

st.divider()

# ── Shared add-on genes ───────────────────────────────────────────────────────
st.subheader(f"Shared add-on genes · {n_addon}")
if n_addon:
    st.caption("Custom genes present in every sample — included in the consensus.")
    st.markdown(_chips(con["addon"], bg="#E6F4EA", colour="#0A5C3E"), unsafe_allow_html=True)
else:
    st.info("No custom add-on genes are shared across all samples; the consensus is the base panel.")

# ── Excluded add-on genes ─────────────────────────────────────────────────────
if con["excluded_addon"]:
    st.subheader(f"Excluded add-on genes · {len(con['excluded_addon'])}")
    st.caption("Custom genes present in **some but not all** samples — dropped from the "
               "consensus so they can't create a panel artifact. The matrix shows which "
               "samples carry each one.")
    presence = con["presence"]
    rows = []
    for g in con["excluded_addon"]:
        row = {"gene": g}
        row.update({sid: ("✓" if presence[g][sid] else "·") for sid in con["slides"]})
        row["in / total"] = f"{sum(presence[g].values())} / {n_samples}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.caption("✅ No partially-shared add-on genes — every sample carries the same custom panel.")

# ── Per-sample contribution ───────────────────────────────────────────────────
st.divider()
st.subheader("Per-sample panel")
ps = con["per_slide"]
cond_by_id = {s["slide_id"]: s.get("condition", "") for s in valid}
tbl = pd.DataFrame([
    {"Sample": sid, "Condition": cond_by_id.get(sid, ""),
     "Total genes": ps[sid]["total"], "Base": ps[sid]["base"], "Custom": ps[sid]["custom"],
     "Custom in consensus": ps[sid]["custom_in_consensus"],
     "Custom dropped": ps[sid]["custom"] - ps[sid]["custom_in_consensus"]}
    for sid in con["slides"]
])
st.dataframe(tbl, use_container_width=True, hide_index=True)

# ── Persist / activate ────────────────────────────────────────────────────────
st.divider()
out_dir = Path(st.session_state["output_dir"])
con_path = out_dir / "consensus_panel.json"

payload = {
    "generated_at": datetime.now().isoformat(),
    "n_samples": n_samples, "slides": con["slides"],
    "n_consensus": n_total, "n_base": n_base, "n_addon": n_addon,
    "base_genes": con["base"], "addon_genes": con["addon"],
    "consensus_genes": con["consensus"],
    "excluded_addon": con["excluded_addon"], "excluded_base": con["excluded_base"],
}

dl1, dl2 = st.columns(2)
dl1.download_button("⬇️ Consensus panel (JSON)", data=json.dumps(payload, indent=2),
                    file_name="consensus_panel.json", mime="application/json",
                    use_container_width=True)
dl2.download_button("⬇️ Consensus genes (CSV)",
                    data=pd.DataFrame({"gene": con["consensus"],
                                       "panel": (["base"] * n_base + ["add-on"] * n_addon)}
                                      ).to_csv(index=False),
                    file_name="consensus_panel.csv", mime="text/csv",
                    use_container_width=True)

active = st.session_state.get("panel_mode") == "consensus"
already = con_path.exists()
st.markdown("#### Use this panel")
st.caption("Locks the consensus as the active gene set for Sample PCA, clustering and all "
           "downstream quantification. (It is the default; this also writes a record to the "
           "output directory.)")
if st.button("✅ Use this consensus panel", type="primary", use_container_width=True):
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        con_path.write_text(json.dumps(payload, indent=2))
        st.session_state["panel_mode"] = "consensus"
        st.success(f"Consensus panel active ({n_total} genes) and saved to `{con_path}`.")
        st.rerun()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not save the consensus panel: {e}")
elif active and already:
    st.success(f"✅ Consensus panel is active ({n_total} genes) · record at `{con_path}`.")
elif active:
    st.info("Consensus mode is active (default). Click above to also save a record to the "
            "output directory.")

st.info("Because the consensus is a strict intersection, **no gene is ever zero-filled** — "
        "which removes the main way a differing add-on panel could fake a differential-"
        "expression hit. Advanced panel modes (partial-union / union) remain available via "
        "the study config, but reintroduce zero-filling for partially-shared genes.")
