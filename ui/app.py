"""Streamlit UI -- Cloud #1 (deployed on Streamlit Community Cloud).

This is a THIN client:
  * every training job and every prediction is an HTTPS call to the FastAPI
    service (no model code, no torch import here),
  * the ONLY database access is a read-only anon-key query against the `runs`
    table for the 'Run History' tab (no SQL writes here).

Grad-CAM images arrive from the API already rendered as base64 PNGs; the UI just
decodes and displays them -- no image processing libraries needed here.

Configuration comes from st.secrets (see .streamlit/secrets.toml.example):
    API_URL              -> your Render.com base URL
    SUPABASE_URL         -> https://<ref>.supabase.co
    SUPABASE_ANON_KEY    -> the public anon key (safe to ship to the browser)
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from supabase import create_client

sys.path.insert(0, str(Path(__file__).resolve().parent))
import concepts as concepts_math  # noqa: E402
import recent  # noqa: E402
import tryit  # noqa: E402
import evidence  # noqa: E402  # sibling module: loaders for the saved official-supabase-1 evidence

API_URL = st.secrets["API_URL"].rstrip("/")

st.set_page_config(page_title="See-Sense", page_icon="🔍", layout="wide")
st.title("🔍 See-Sense — A Live CNN Classifier with Grad-CAM")
st.caption("Streamlit (this UI) → FastAPI (CNN + Grad-CAM) → Supabase (data). Three clouds.")


@st.cache_resource
def supabase_anon():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])


def api_get(path: str, **params):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=180)
    r.raise_for_status()
    return r.json()


concepts, try_tab, compare_tab, recent_tab, card_tab = st.tabs(
    ["Concepts", "Try It", "Compare Models", "Recent Classifications", "Model Card"]
)

# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------
def _trace_table(layers, n):
    return pd.DataFrame(
        [
            {"Layer": t["layer"], "Output size": f"{t['size']}×{t['size']}",
             "Receptive field r": f"{t['r']}×{t['r']}", "Jump j": t["j"]}
            for t in concepts_math.trace(layers, n)
        ]
    )


with concepts:
    st.header("How a CNN sees: convolution, receptive fields and Grad-CAM")
    st.caption(
        "Every table on this page is computed by the same functions that are unit-tested, "
        "using this project's own network."
    )

    # ---- 1. Convolution ---------------------------------------------------
    st.subheader("1. Convolution")
    st.markdown(
        "A convolutional layer slides a small learnable grid of weights, the **kernel** $K$ "
        "(here 3×3), over the image $X$. At each position it multiplies the kernel with the "
        "pixels underneath and adds them up, giving one number of the **feature map** $A$ "
        "(plus a learned bias $b$):"
    )
    st.latex(r"A_{ij} = b + \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} K_{mn}\, X_{\,si+m,\;sj+n}")
    st.markdown(
        "The same weights are reused at every position, so the layer detects one pattern "
        "wherever it appears. A layer with several kernels produces several feature maps "
        "(*channels*). A ReLU keeps positive responses, and **max-pooling** keeps the largest "
        "value in each 2×2 window, halving the width and height. Finally a linear layer turns "
        "the last feature maps into four class scores, trained with cross-entropy."
    )

    # ---- 2. Stride, padding, output size ----------------------------------
    st.subheader("2. Stride, padding and output size")
    st.markdown(
        "**Stride** $s$ is how many pixels the kernel jumps between positions. **Padding** $p$ "
        "adds a border of zeros around the input. For input width $n$ and kernel size $k$:"
    )
    st.latex(r"n_{\text{out}} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1")
    st.markdown("**Worked example** with a 28-pixel-wide input and a 3×3 kernel:")
    ex = [(28, 3, 1, 0), (28, 3, 1, 1), (28, 3, 2, 1)]
    st.dataframe(
        pd.DataFrame(
            [
                {"n": n, "k": k, "stride s": s, "padding p": p,
                 "Calculation": f"⌊({n} + {2 * p} − {k}) / {s}⌋ + 1",
                 "Output": concepts_math.out_size(n, k, s, p)}
                for n, k, s, p in ex
            ]
        ),
        use_container_width=True, hide_index=True,
    )
    st.markdown(
        "With no padding the map shrinks by $k-1$ pixels (28 → 26). **Padding 1 with a 3×3 "
        "kernel keeps the size** (28 → 28), which is what this project's CNN uses. Stride 2 "
        "roughly halves it (28 → 14). Max-pooling follows the same formula with $k=2,\\ s=2,\\ p=0$: "
        "28 → 14 → 7."
    )
    with st.expander("Try your own numbers"):
        cc1, cc2, cc3, cc4 = st.columns(4)
        cn = cc1.number_input("Input size n", 1, 512, 28)
        ck = cc2.number_input("Kernel k", 1, 15, 3)
        cs = cc3.number_input("Stride s", 1, 8, 1)
        cp = cc4.number_input("Padding p", 0, 8, 1)
        try:
            st.success(f"Output size = ⌊({cn} + {2 * cp} − {ck}) / {cs}⌋ + 1 = "
                       f"**{concepts_math.out_size(int(cn), int(ck), int(cs), int(cp))}**")
        except ValueError:
            st.warning("The kernel is larger than the padded input, so no output positions fit.")

    st.markdown("**This project's CNN** (28×28 grayscale input, 4 classes):")
    st.dataframe(_trace_table(concepts_math.OUR_CNN, 28), use_container_width=True, hide_index=True)
    st.caption(
        f"The final 16 × 7 × 7 feature maps feed a linear layer with 16·7·7·4 + 4 = 3,140 weights. "
        f"In total the baseline has {concepts_math.our_cnn_param_count():,} parameters, matching the "
        "4,388 recorded in Supabase."
    )

    # ---- 3. Receptive field ----------------------------------------------
    st.subheader("3. Receptive field")
    st.markdown(
        "The **receptive field** $r$ of a feature-map cell is the patch of the original image "
        "that can influence it. Track two numbers through the layers: $r$, and the **jump** $j$, "
        "the distance in input pixels between neighbouring cells. Start with $r_0 = 1,\\ j_0 = 1$."
    )
    st.latex(r"r_{\ell} = r_{\ell-1} + (k_{\ell}-1)\, j_{\ell-1}, \qquad j_{\ell} = j_{\ell-1}\, s_{\ell}")
    st.markdown(
        "*Derivation.* Moving the kernel one step in layer $\\ell$ shifts the window by "
        "$j_{\\ell-1}$ input pixels. A kernel of size $k$ spans $k-1$ such steps beyond its first "
        "cell, so it widens the field by $(k-1)\\,j_{\\ell-1}$. Stride multiplies the jump because "
        "each output step now covers $s$ input steps."
    )
    st.markdown("**Worked example: three stacked 3×3 convolutions, stride 1.** Here $j$ stays 1:")
    st.latex(
        r"r_1 = 1 + 2\cdot 1 = 3,\qquad r_2 = 3 + 2\cdot 1 = 5,\qquad r_3 = 5 + 2\cdot 1 = 7"
    )
    st.dataframe(_trace_table(concepts_math.THREE_3X3, 28), use_container_width=True, hide_index=True)
    st.markdown(
        "Each extra 3×3 layer adds 2 pixels, so **3 → 5 → 7**: three small layers see a 7×7 patch, "
        "like a single 7×7 kernel but with fewer weights per channel pair (3·9 = 27 versus 49)."
    )
    st.markdown("**In this project's CNN**, pooling doubles the jump, so growth speeds up:")
    st.latex(
        r"3 \xrightarrow{\text{pool}} 3+1\cdot 1 = 4 \xrightarrow{\text{conv}} 4+2\cdot 2 = 8"
        r" \xrightarrow{\text{pool}} 8+1\cdot 2 = 10"
    )
    st.markdown(
        "Each of the 7×7 cells Grad-CAM works with therefore looks at a **10×10-pixel patch** of "
        "the 28×28 image, and neighbouring cells are 4 pixels apart. This is why Grad-CAM maps "
        "are coarse."
    )

    # ---- 4. Grad-CAM -------------------------------------------------------
    st.subheader("4. Grad-CAM: where did the network look?")
    st.markdown(
        "Let $A^k$ be the $k$-th feature map of the last convolutional block (here 16 maps of "
        "7×7) and $y^c$ the network's score for class $c$ (before softmax)."
    )
    st.markdown(
        "**Step 1 — how much does each map matter?** The gradient $\\partial y^c / \\partial A^k_{ij}$ "
        "says how much the class score would change if cell $(i,j)$ of map $k$ were larger. "
        "Average it over all $Z = 7\\times 7$ cells to get one weight per map:"
    )
    st.latex(r"\alpha_k^c = \frac{1}{Z}\sum_i\sum_j \frac{\partial y^c}{\partial A^k_{ij}}")
    st.markdown(
        "**Step 2 — combine.** The map-weighted sum shows where evidence for class $c$ is "
        "located. ReLU keeps only places that *raise* the score:"
    )
    st.latex(r"L^c_{\text{Grad-CAM}} = \mathrm{ReLU}\!\left(\sum_k \alpha_k^c\, A^k\right)")
    st.markdown(
        "**Step 3 — display.** Divide by the maximum so values lie in [0, 1], upsample the 7×7 "
        "map to 28×28 and overlay it in red. In the app, $c$ is the class the model predicted."
    )
    st.markdown(
        "**Why this works here.** After the last feature maps this network has only a linear "
        "layer, so $y^c = \\sum_{k,i,j} w^c_{kij} A^k_{ij} + b^c$ and the gradient is simply the "
        "weight $w^c_{kij}$. Then $\\alpha_k^c$ is the average weight the class gives map $k$, and "
        "$\\alpha_k^c A^k_{ij}$ is that map's contribution at cell $(i,j)$ to the score."
    )
    st.warning(
        "**Limitations.** Grad-CAM is coarse: its resolution is 7×7 and each cell sees a "
        "10×10 patch. It shows regions *associated with* a class score; it does not prove the "
        "model's reasoning, and a tidy-looking heatmap can sit on a wrong prediction (see the "
        "Compare Models gallery, where example 924 was wrong with 95% confidence). Averaging "
        "gradients and ReLU discard negative evidence and fine detail, and the map depends on "
        "the chosen layer and class. Treat it as a clue, not an explanation."
    )

# ---------------------------------------------------------------------------
# Try It  (predictions go through the API; the UI never writes to Supabase)
# ---------------------------------------------------------------------------
def _show_result(resp: dict):
    st.success(f"Predicted: **{resp['class_name']}**  (confidence {resp['confidence']:.2%})")
    if resp.get("true_class"):
        verdict = "matches" if resp["true_class"] == resp["class_name"] else "does not match"
        st.caption(f"Ground truth for this generated sample: {resp['true_class']} "
                   f"(the prediction {verdict}).")
    ic, cc = st.columns(2)
    ic.image(tryit.decode_png(resp["input_png"]), caption="Input (28×28, enlarged)", width=224)
    cc.image(tryit.decode_png(resp["cam_png"]), caption="Grad-CAM (red = most influential)", width=224)
    st.caption(f"Image sha256 {resp['image_sha256'][:16]}… logged to Supabase (no pixels stored).")


with try_tab:
    st.header("Try It")
    st.info(
        "This model recognizes **four synthetic shapes** — horizontal bar, vertical bar, "
        "diagonal line and filled block — on small noisy 28×28 grayscale images. It is "
        "**not a general photo classifier**: a photo of a cat or a face will still get one "
        "of those four labels, and that label will be meaningless."
    )

    try:
        default_run = tryit.resolve_default_run()
    except tryit.TryItError as exc:
        default_run = None
        st.warning(f"Try It is unavailable. {exc}")

    if default_run:
        st.caption(
            f"Model: `{default_run['run_name']}` from experiment group "
            f"`{default_run['group']}` (run ID {default_run['run_id']}, taken from the saved "
            "experiment records). It is the project's full improved model, chosen in advance "
            "rather than by looking at test scores."
        )
        st.caption(
            "Each prediction is logged to Supabase as an image hash plus the predicted class "
            "and confidence — never the pixels. Press a button once and wait: predictions "
            "are not retried automatically."
        )

        if "api_ready" not in st.session_state:
            with st.spinner("Checking the prediction service (it may be waking up)..."):
                st.session_state["api_ready"] = tryit.wake_api(API_URL)
        if not st.session_state["api_ready"]:
            st.warning(
                "The prediction service hasn't responded yet. Free-tier services sleep when "
                "idle and can take up to a minute to start. You can still press a button below."
            )
            if st.button("Check again"):
                st.session_state.pop("api_ready", None)
                st.rerun()

        run_id = default_run["run_id"]
        st.subheader("Option A — upload an image")
        up = st.file_uploader("PNG / JPG (best: a small, simple shape on a plain background)",
                              type=["png", "jpg", "jpeg"])
        if up is not None and st.button("Classify upload", type="primary"):
            try:
                with st.spinner("Running the CNN + Grad-CAM..."):
                    resp = tryit.predict_upload(API_URL, run_id, up.name, up.getvalue(), up.type)
                _show_result(resp)
            except tryit.TryItError as exc:
                st.error(str(exc))

        st.subheader("Option B — classify a generated sample")
        shape_names = ["horizontal", "vertical", "diagonal", "block"]
        c1, c2 = st.columns(2)
        shape = c1.selectbox("Shape", ["random"] + shape_names)
        seed = c2.number_input("Sample seed", value=0, min_value=0, step=1)
        if st.button("Generate & classify"):
            try:
                with st.spinner("Generating a sample and running Grad-CAM..."):
                    resp = tryit.predict_sample(
                        API_URL, run_id, int(seed),
                        None if shape == "random" else shape_names.index(shape),
                    )
                _show_result(resp)
            except tryit.TryItError as exc:
                st.error(str(exc))

# ---------------------------------------------------------------------------
# Recent Classifications  (read-only anon-key query of the recent_classifications view)
# ---------------------------------------------------------------------------
with recent_tab:
    st.header("Recent Classifications")
    st.caption(
        "Read directly from Supabase with the anon key (a read-only view with no image "
        f"hashes). Shows the latest available classifications, **up to {recent.LIMIT}** — "
        "not all-time results."
    )
    if st.button("Refresh") or "recent_rows" not in st.session_state:
        with st.spinner("Loading recent classifications..."):
            try:
                st.session_state["recent_rows"] = recent.fetch_recent(supabase_anon)
                st.session_state["recent_error"] = None
            except recent.RecentError as exc:
                st.session_state["recent_rows"] = None
                st.session_state["recent_error"] = str(exc)

    if st.session_state.get("recent_error"):
        st.error(st.session_state["recent_error"])
    elif st.session_state.get("recent_rows") is not None:
        rows, skipped = recent.clean_rows(st.session_state["recent_rows"])
        if not rows:
            st.info("No classifications yet. Try the Try It tab, then press Refresh.")
        else:
            st.markdown(f"**{len(rows)} classification{'s' if len(rows) != 1 else ''} shown**"
                        + (f" ({skipped} malformed row{'s' if skipped != 1 else ''} skipped)"
                           if skipped else ""))
            counts = recent.class_counts(rows)
            st.subheader(f"Class distribution (latest {len(rows)}, up to {recent.LIMIT})")
            st.bar_chart(pd.DataFrame({"Classifications": counts}))
            st.subheader("Latest classifications")
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        min_value=0.0, max_value=1.0, format="%.2f"
                    ),
                },
            )

# ---------------------------------------------------------------------------
# Compare Models + Grad-CAM gallery (saved official-supabase-1 evidence)
# ---------------------------------------------------------------------------
def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _show_figure(key: str, caption: str):
    path = evidence.FIGURES[key]
    if path.is_file():
        st.image(str(path), caption=caption, use_column_width=True)
    else:
        st.warning(f"Figure not available: {path.name}")


def _section(title: str, loader, render):
    """Run one evidence section; a failure shows a message, not a traceback."""
    st.subheader(title)
    try:
        render(loader())
    except evidence.EvidenceError as exc:
        st.warning(f"This section is unavailable. {exc}")
    except Exception:  # noqa: BLE001
        st.warning("This section is unavailable: the saved evidence could not be read.")


def _render_summary(rows):
    params = {}
    try:
        params = evidence.load_param_counts()
    except evidence.EvidenceError:
        pass
    df = pd.DataFrame(
        {
            "Variant": r["variant"],
            "Seeds": r["n_seeds"],
            "Test accuracy (mean ± sd)": f"{_pct(r['mean_test_accuracy'])} ± {r['sd_test_accuracy'] * 100:.2f}",
            "Macro-F1 (mean ± sd)": f"{r['mean_test_macro_f1']:.4f} ± {r['sd_test_macro_f1']:.4f}",
            "Test loss (mean)": f"{r['mean_test_loss']:.4f}",
            "Params": params.get(r["variant"], "n/a"),
            "Train s (mean)": f"{r['mean_train_seconds']:.1f}",
        }
        for r in rows
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(
        "sd = sample standard deviation across seeds. With five seeds these are "
        "rough estimates; baseline and no_batchnorm vary a lot between seeds."
    )
    _show_figure("test_metrics_by_seed", "Test metrics for every variant and seed")


def _render_audit(audit):
    if audit["passed"]:
        st.success(
            "Matched-configuration audit: PASS. Each of the 5 seeds has 6 runs, and "
            "within a seed all runs share one dataset, data seed, split, split hash, "
            "split sizes, noise, epoch budget, batch size, initial LR, optimizer and "
            "augmentation shift."
        )
    else:
        st.error("Matched-configuration audit: FAIL. " + "; ".join(audit["problems"]))
    with st.expander("Audit query output (query 1)"):
        st.dataframe(pd.DataFrame(audit["rows"]), use_container_width=True, hide_index=True)


def _render_paired(rows):
    st.markdown(
        "Each ablation removes **one** technique from `full_improved` while the other "
        "techniques stay enabled, then compares the two on the **same seed**. The "
        "difference is therefore that technique's contribution *given* the others, "
        "not its stand-alone effect. (`baseline` removes the architecture changes "
        "and all four techniques together.) Negative = lower than the full model."
    )
    df = pd.DataFrame(
        {
            "Variant": r["variant"],
            "Technique removed": r["technique_removed"],
            "Δ accuracy per seed (pp)": ", ".join(f"{a:+.2f}" for a in r["per_seed_acc_pp"]),
            "Mean Δ acc (pp)": f"{r['mean_acc_pp']:+.2f}",
            "95% CI (pp, paired t)": f"[{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}]"
            + (" includes 0" if r["ci_includes_zero"] else ""),
            "Mean Δ macro-F1": f"{r['mean_f1']:+.4f}",
            "Mean Δ test loss": f"{r['mean_loss']:+.4f}",
            "Seeds lower than full": f"{r['seeds_lower']} of {r['n_seeds']}",
        }
        for r in rows
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.info(
        "Reading this table: removing augmentation lowered accuracy on all 5 seeds "
        "(interval excludes 0). For dropout and the LR scheduler the intervals include "
        "0 and the seeds disagree in sign, so these data do **not** show that either "
        "clearly improved accuracy. The batch-norm interval also includes 0 because "
        "seed variance is huge, even though all 5 seeds scored lower without it. "
        "Intervals come from only 5 seeds."
    )


def _render_gradcam(examples):
    source = {}
    try:
        source = evidence.load_gradcam_source()
    except evidence.EvidenceError:
        pass
    if source:
        st.caption(
            f"Source: run `{source.get('run_name')}` (Supabase run id {source.get('run_id')}, "
            f"dataset id {source.get('dataset_id')}), regenerated test accuracy "
            f"{source.get('regenerated_accuracy', 0):.2%} matching the stored "
            f"{source.get('stored_accuracy', 0):.2%}; {source.get('n_correct')} correct and "
            f"{source.get('n_incorrect')} incorrect of {source.get('n_test')} test images."
        )
    interp = None
    try:
        interp = evidence.load_failure_interpretations()
    except evidence.EvidenceError as exc:
        st.warning(f"Failure interpretations unavailable. {exc}")

    for set_name, label, fig in (
        ("correct", "Correct predictions (5)", "gradcam_correct"),
        ("incorrect", "Incorrect predictions (8)", "gradcam_incorrect"),
    ):
        st.markdown(f"#### {label}")
        _show_figure(fig, f"Grad-CAM, {set_name} examples (left to right = table order below)")
        subset = [e for e in examples if e["set"] == set_name]
        df = pd.DataFrame(
            {
                "Test index": e["test_index"],
                "True": e["true"],
                "Predicted": e["predicted"],
                "Confidence": f"{e['confidence']:.1%}",
                "Note": f"Correct but uncertain ({e['confidence']:.0%} confidence)" if e["uncertain"] else "",
            }
            for e in subset
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("#### Failure interpretations")
    st.caption("Two of the eight incorrect examples have a detailed written analysis.")
    if interp:
        for idx, item in interp["examples"].items():
            with st.expander(f"Example {idx}: {item['title']}", expanded=True):
                st.markdown(item["text"])
        if interp["limits"]:
            st.markdown("**Interpretation limits**")
            st.markdown(interp["limits"])


with compare_tab:
    st.header("Compare Models")
    st.info(
        f"**Saved experiment results** — group `{evidence.GROUP}`, 5 seeds (0–4) × 6 "
        "variants = 30 runs, one fixed 2,000-image synthetic dataset. Tables come from "
        "the official Supabase SQL exports (queries 1–7, saved as CSV in "
        "`experiments/results/official-supabase-1-sql/`); figures were generated from "
        "those exports. This tab displays files saved in the repository; it does not "
        "re-query Supabase or retrain."
    )
    gone = evidence.missing_figures()
    if gone:
        st.warning("Some saved figures are missing: " + ", ".join(gone))

    _section("Six-variant results", evidence.load_variant_summary, _render_summary)
    _section("Matched-configuration audit", evidence.load_audit, _render_audit)
    _section("Paired ablation differences vs full_improved", evidence.load_paired_differences, _render_paired)

    st.subheader("Training and validation loss curves")
    _show_figure("loss_curves", "Loss curves by variant (official-supabase-1)")
    st.subheader("Confusion matrices")
    _show_figure("confusion_matrices", "Test confusion matrices by variant, summed over seeds")

    st.divider()
    st.header("Grad-CAM gallery")
    _section("Correct and incorrect examples (full_improved, seed 0)", evidence.load_gradcam_examples, _render_gradcam)

# ---------------------------------------------------------------------------
# Model Card
# ---------------------------------------------------------------------------
with card_tab:
    try:
        st.markdown(evidence.load_model_card())
    except evidence.EvidenceError as exc:
        st.warning(f"The model card is unavailable. {exc}")
