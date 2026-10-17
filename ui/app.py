"""Streamlit UI -- Cloud #1 (deployed on Streamlit Community Cloud).

This is a THIN client:
  * every training job and every prediction is an HTTPS call to the FastAPI
    service (no model code, no torch import here),
  * the ONLY database access is read-only anon-key queries (the `recent_predictions` and
    `recent_classifications` views and the `runs` table for the drill-down; no SQL writes here).

Grad-CAM images arrive from the API already rendered as base64 PNGs; the UI just
decodes and displays them -- no image processing libraries needed here.

Configuration comes from st.secrets (see .streamlit/secrets.toml.example):
    API_URL              -> your Render.com base URL
    SUPABASE_URL         -> https://<ref>.supabase.co
    SUPABASE_ANON_KEY    -> the public anon key (safe to ship to the browser)
"""
from __future__ import annotations

import base64
import logging
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
import evidence  # noqa: E402  # sibling module: loaders for the saved experiment evidence

API_URL = st.secrets["API_URL"].rstrip("/")

st.set_page_config(page_title="See-Sense", page_icon="🔍", layout="wide")
st.title("🔍 See-Sense — Fashion-MNIST CNN Classifier with Grad-CAM")
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
        "the last feature maps into ten class scores (one per Fashion-MNIST category), trained with cross-entropy."
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

    st.markdown("**This project's CNN** (28×28 grayscale input, 10 classes):")
    st.dataframe(_trace_table(concepts_math.OUR_CNN, 28), use_container_width=True, hide_index=True)
    st.caption(
        f"The final 16 × 7 × 7 feature maps feed a linear layer with 16·7·7·10 + 10 = 7,850 weights. "
        f"In total the baseline has {concepts_math.our_cnn_param_count():,} parameters (computed from "
        "the layer sizes; the training code counts the same number on the PyTorch model)."
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
        "map to 28×28 and overlay it in red. In the app, $c$ is the class the model predicted, and "
        "$y^c$ is the raw class score (logit), so the temperature used to calibrate probabilities "
        "never changes the heatmap."
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
        "model's reasoning, and a tidy-looking heatmap can sit on a wrong prediction (the Compare "
        "Models gallery shows confident mistakes alongside correct examples). Averaging "
        "gradients and ReLU discard negative evidence and fine detail, and the map depends on "
        "the chosen layer and class. Treat it as a clue, not an explanation."
    )

# ---------------------------------------------------------------------------
# Try It  (predictions go through the API; the UI never writes to Supabase)
# ---------------------------------------------------------------------------
def _show_result(resp: dict, truth: str | None = None):
    st.success(f"Predicted: **{resp['class_name']}**  (confidence {resp['confidence']:.2%})")
    if truth:
        verdict = "matches" if truth == resp["class_name"] else "does not match"
        st.caption(f"Ground truth for this held-out example: {truth} (the prediction {verdict}).")
    if resp.get("calibrated"):
        st.caption(
            f"Probabilities are divided by a temperature of {resp.get('temperature', 1.0):.3f} that was fitted "
            "on validation data. That is a calibration *method*; how well it worked is measured on the "
            "Compare Models tab, not assumed here."
        )
    else:
        st.caption("These are plain softmax probabilities (this checkpoint has no calibration).")
    rows = tryit.top5_rows(resp)
    left, mid, right = st.columns([1.1, 1, 1])
    with left:
        st.markdown("**Top five probabilities**")
        if rows:
            st.bar_chart(pd.DataFrame(rows).set_index("Class")["Probability"], horizontal=True)
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                         column_config={"Probability": st.column_config.ProgressColumn(
                             min_value=0.0, max_value=1.0, format="%.3f")})
        else:
            st.warning("The service did not return a probability list.")
    mid.image(tryit.decode_png(resp["input_png"]), caption="Input (28×28, enlarged)", width=224)
    right.image(tryit.decode_png(resp["cam_png"]), caption="Grad-CAM for the predicted class (red = most influential)", width=224)
    st.caption("A plain image with no coloured overlay means Grad-CAM found no positive evidence for the predicted "
               "class. This is common for **Dress** and **Coat** predictions with this model (see the Model Card), "
               "so a blank map is not a bug in your upload.")
    if resp.get("logged") is True:
        st.caption(f"Logged to Supabase as prediction #{resp.get('prediction_id')}: image hash {resp['image_sha256'][:16]}…, "
                   "label and probabilities. No pixels are stored.")
    else:
        st.warning("The service did not confirm that this prediction was logged.")


with try_tab:
    st.header("Try It")
    st.info(
        "This model recognizes the **ten Fashion-MNIST clothing categories**: T-shirt/top, trouser, "
        "pullover, dress, coat, sandal, shirt, sneaker, bag and ankle boot. It was trained on small "
        "28×28 grayscale pictures of a single light-coloured item on a dark background. It is "
        "**not a general photo classifier**: a photo of a cat, a face or a cluttered room will still get "
        "one of those ten labels, and that label will be meaningless. Dark items on a light "
        "background are *not* inverted for you and will usually be misread."
    )

    served = None
    try:
        served = tryit.resolve_served_run()
    except tryit.TryItError as exc:
        st.warning(f"Try It is unavailable. {exc}")

    if served:
        if "api_ready" not in st.session_state:
            with st.spinner("Checking the prediction service (it may be waking up)..."):
                st.session_state["api_ready"] = tryit.wake_api(API_URL)
        if not st.session_state["api_ready"]:
            st.warning(
                "The prediction service hasn't responded yet. Free-tier services sleep when "
                "idle and can take up to a minute to start."
            )
            if st.button("Check again"):
                st.session_state.pop("api_ready", None)
                st.session_state.pop("served_verified", None)
                st.rerun()
        elif "served_verified" not in st.session_state:
            try:
                tryit.verify_served_run(API_URL, served)
                st.session_state["served_verified"] = True
            except tryit.TryItError as exc:
                st.session_state["served_verified"] = str(exc)
        verified = st.session_state.get("served_verified")

        if verified is True:
            st.caption(
                f"Model: `{served['run_name']}` from experiment group `{served['group']}` (run ID "
                f"{served['run_id']}). Chosen in advance in `config/served_model.json`, matched to the saved "
                "experiment records and verified against the prediction service; not picked by test score."
            )
            st.caption(
                "Each prediction is logged to Supabase as an image hash plus the predicted class and its "
                "probability vector, never the pixels. Press a button once and wait: predictions are not "
                "retried automatically. If logging fails, you get an error, not a result."
            )
            run_id = served["run_id"]
            st.subheader("Option A — upload an image")
            st.warning(tryit.UPLOAD_WARNING)
            st.caption(tryit.UPLOAD_SAMPLES_HINT)
            up = st.file_uploader("PNG / JPG (best: one light-coloured clothing item on a dark background)",
                                  type=["png", "jpg", "jpeg"])
            if up is not None and st.button("Classify upload", type="primary"):
                try:
                    with st.spinner("Running the CNN + Grad-CAM..."):
                        resp = tryit.predict_upload(API_URL, run_id, up.name, up.getvalue(), up.type)
                    _show_result(resp)
                except tryit.TryItError as exc:
                    st.error(str(exc))

            st.subheader("Option B — classify a held-out clothing example")
            samples = tryit.load_samples()
            if not samples:
                st.info("No held-out example images are bundled with this deployment.")
            else:
                st.caption(
                    "These are official Fashion-MNIST **test** images (the first two of each class), "
                    "which no model in this project trained or validated on. They are sent through the "
                    "same upload path as your own images."
                )
                by_class: dict[str, list[dict]] = {}
                for item in samples["samples"]:
                    by_class.setdefault(item["class_name"], []).append(item)
                c1, c2 = st.columns(2)
                cls = c1.selectbox("Category", list(by_class))
                pick = c2.selectbox("Example", [f"{cls} #{i + 1} (test image {it['test_index']})"
                                                for i, it in enumerate(by_class[cls])])
                item = by_class[cls][int(pick.split("#")[1].split(" ")[0]) - 1]
                path = tryit.SAMPLES_DIR / item["file"]
                st.image(str(path), caption="Selected example (28×28, enlarged)", width=112)
                if st.button("Classify this example"):
                    try:
                        with st.spinner("Running the CNN + Grad-CAM..."):
                            resp = tryit.predict_upload(API_URL, run_id, item["file"], path.read_bytes(), "image/png")
                        _show_result(resp, truth=item["class_name"])
                    except tryit.TryItError as exc:
                        st.error(str(exc))
        elif isinstance(verified, str):
            st.error(verified)
            if st.button("Verify again"):
                st.session_state.pop("served_verified", None)
                st.rerun()

# ---------------------------------------------------------------------------
# Recent Classifications  (read-only anon-key queries; drill-down to the producing run)
# ---------------------------------------------------------------------------
with recent_tab:
    st.header("Recent Classifications")
    st.caption(
        "Read directly from Supabase with the anon key through read-only views that contain no image "
        f"hashes. Shows the latest available classifications, **up to {recent.LIMIT}** — not all-time results."
    )
    if st.button("Refresh") or "recent_pred_rows" not in st.session_state:
        with st.spinner("Loading recent classifications..."):
            st.session_state.pop("drill_run", None)
            try:
                st.session_state["recent_pred_rows"] = recent.fetch_recent_predictions(supabase_anon)
                st.session_state["recent_error"] = None
                st.session_state["recent_fallback"] = False
            except recent.RecentError as exc:
                # The predictions view comes with migration 004. Fall back to the older, run-less list.
                try:
                    st.session_state["recent_pred_rows"] = recent.fetch_recent(supabase_anon)
                    st.session_state["recent_error"] = None
                    st.session_state["recent_fallback"] = True
                except recent.RecentError:
                    st.session_state["recent_pred_rows"] = None
                    st.session_state["recent_error"] = str(exc)

    if st.session_state.get("recent_error"):
        st.error(st.session_state["recent_error"])
    elif st.session_state.get("recent_pred_rows") is not None:
        fallback = st.session_state.get("recent_fallback")
        if fallback:
            st.warning("The predictions view isn't available yet, so this shows the older classification "
                       "list without probabilities or run details.")
            rows, skipped = recent.clean_rows(st.session_state["recent_pred_rows"])
        else:
            rows, skipped = recent.clean_prediction_rows(st.session_state["recent_pred_rows"])
        if not rows:
            st.info("No classifications yet. Try the Try It tab, then press Refresh.")
        else:
            st.markdown(f"**{len(rows)} classification{'s' if len(rows) != 1 else ''} shown**"
                        + (f" ({skipped} malformed row{'s' if skipped != 1 else ''} skipped)" if skipped else ""))
            counts = recent.class_counts(rows)
            st.subheader(f"Class distribution (latest {len(rows)}, up to {recent.LIMIT})")
            st.bar_chart(pd.DataFrame({"Classifications": counts}))
            st.subheader("Latest classifications")
            table = pd.DataFrame([{k: v for k, v in r.items() if k != "Probabilities"} for r in rows])
            st.dataframe(
                table, use_container_width=True, hide_index=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
                    "Run test accuracy": st.column_config.NumberColumn(format="%.4f"),
                },
            )
            if not fallback:
                st.subheader("Drill down: which run produced a classification?")
                labels = [f"{r['Time']} · {r['Class']} ({r['Confidence']:.0%}) · run {r['Run ID']}" for r in rows]
                chosen = st.selectbox("Pick a classification", range(len(rows)), format_func=lambda i: labels[i])
                row = rows[chosen]
                left, right = st.columns(2)
                with left:
                    st.markdown("**Top five probabilities logged for it**")
                    st.dataframe(pd.DataFrame(recent.top_k(row["Probabilities"])), hide_index=True,
                                 use_container_width=True,
                                 column_config={"Probability": st.column_config.ProgressColumn(
                                     min_value=0.0, max_value=1.0, format="%.3f")})
                with right:
                    st.markdown(f"**Producing run {row['Run ID']}**")
                    key = ("drill_run", row["Run ID"])
                    if st.session_state.get("drill_run_key") != key:
                        try:
                            st.session_state["drill_run"] = recent.fetch_run(supabase_anon, row["Run ID"])
                            st.session_state["drill_err"] = None
                        except recent.RecentError as exc:
                            st.session_state["drill_run"], st.session_state["drill_err"] = None, str(exc)
                        st.session_state["drill_run_key"] = key
                    if st.session_state.get("drill_err"):
                        st.error(st.session_state["drill_err"])
                    elif st.session_state.get("drill_run") is None:
                        st.info("That run's record is no longer available.")
                    else:
                        st.dataframe(pd.DataFrame(recent.run_summary(st.session_state["drill_run"])),
                                     hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Compare Models + Grad-CAM gallery
# ---------------------------------------------------------------------------
def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _show_figure(ev: "evidence.EvidenceSet", key: str, caption: str):
    path = ev.figure(key)
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
        logging.getLogger("see_sense.ui").exception("evidence section failed: %s", title)  # server log only
        st.warning("This section is unavailable: the saved evidence could not be read.")


def _render_summary(ev, rows):
    params = {}
    try:
        params = evidence.load_param_counts(ev.sql_dir)
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
    st.caption("sd = sample standard deviation across training seeds; with few seeds it is a rough estimate.")
    _show_figure(ev, "test_metrics_by_seed", "Test accuracy for every variant and seed")


def _render_audit(ev, audit):
    n = len(audit["rows"])
    if audit["passed"]:
        st.success(
            f"Matched-configuration audit: PASS. Each of the {n} seeds has 6 runs, and "
            "within a seed all runs share one dataset, split, split hash, split sizes, "
            "epoch budget, batch size, initial LR, optimizer and augmentation shift."
        )
    else:
        st.error("Matched-configuration audit: FAIL. " + "; ".join(audit["problems"]))
    with st.expander("Audit query output (query 1)"):
        st.dataframe(pd.DataFrame(audit["rows"]), use_container_width=True, hide_index=True)


def _render_paired(ev, rows):
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
    return rows


def _render_per_class(ev, rows):
    df = pd.DataFrame(
        {"Variant": r["variant"], "Class": r["class"],
         "Precision": f"{r['precision'] * 100:.1f}% ± {r['precision_sd'] * 100:.1f}",
         "Recall": f"{r['recall'] * 100:.1f}% ± {r['recall_sd'] * 100:.1f}"}
        for r in rows
    )
    _show_figure(ev, "per_class_precision_recall", "Per-class test precision and recall (mean over seeds)")
    with st.expander("Per-class table (mean ± sd over seeds)"):
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_calibration(ev, rows):
    st.markdown(
        "Temperature scaling divides the logits by a single number fitted on **validation** data. "
        "ECE (expected calibration error) and NLL (negative log-likelihood) are lower when the stated "
        "confidences match how often the model is right. The test columns are held-out measurements."
    )
    st.dataframe(pd.DataFrame(
        {"Variant": r["variant"], "Runs": r["n_runs"], "Mean T": f"{r['temperature']:.3f}",
         "Test ECE before → after": f"{r['test_ece_before']:.4f} → {r['test_ece_after']:.4f}",
         "Test NLL before → after": f"{r['test_nll_before']:.4f} → {r['test_nll_after']:.4f}"}
        for r in rows), use_container_width=True, hide_index=True)


def _render_gradcam(ev, examples):
    source = {}
    try:
        source = evidence.load_gradcam_source(ev.gradcam_dir)
    except evidence.EvidenceError:
        pass
    if source:
        st.caption(
            f"Source: run `{source.get('run_name')}` (run id {source.get('run_id')}), regenerated test accuracy "
            f"{source.get('regenerated_accuracy', 0):.2%}"
            + (f" matching the stored {source['stored_accuracy']:.2%}" if "stored_accuracy" in source else "")
            + f"; {source.get('n_correct')} correct and {source.get('n_incorrect')} incorrect of "
            f"{source.get('n_test')} test images."
        )
        if source.get("selection_rule"):
            st.caption("Selection rule (fixed before looking at the figures): " + source["selection_rule"])
    interp = None
    try:
        interp = evidence.load_failure_interpretations(ev.failure_doc)
    except evidence.EvidenceError as exc:
        st.warning(f"Failure interpretations unavailable. {exc}")

    n_by = {k: sum(e["set"] == k for e in examples) for k in ("correct", "incorrect")}
    for set_name, label, fig in (
        ("correct", f"Correct predictions ({n_by['correct']})", "gradcam_correct"),
        ("incorrect", f"Incorrect predictions ({n_by['incorrect']})", "gradcam_incorrect"),
    ):
        st.markdown(f"#### {label}")
        _show_figure(ev, fig, (
            f"Grad-CAM, {set_name} examples: one example per row, top to bottom = table order below. "
            + ("Columns: input, Grad-CAM for the predicted class."
               if set_name == "correct" else
               "Columns: input, Grad-CAM for the predicted class, Grad-CAM for the true class.")
        ))
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
    if interp:
        st.caption(f"{len(interp['examples'])} of the incorrect examples have a written analysis.")
        for idx, item in interp["examples"].items():
            with st.expander(f"Example {idx}: {item['title']}", expanded=True):
                st.markdown(item["text"])
        if interp["limits"]:
            st.markdown("**Interpretation limits**")
            st.markdown(interp["limits"])


def _render_evidence_set(ev: "evidence.EvidenceSet", historical: bool):
    expected = ev.expected_seeds
    _section("Six-variant results", lambda: evidence.load_variant_summary(ev.sql_dir), lambda r: _render_summary(ev, r))
    _section("Matched-configuration audit", lambda: evidence.load_audit(ev.sql_dir, expected), lambda a: _render_audit(ev, a))
    _section("Paired ablation differences vs full_improved",
             lambda: evidence.load_paired_differences(ev.sql_dir, expected), lambda r: _render_paired(ev, r))
    if historical:
        st.info(
            "Reading this historical table: removing augmentation lowered accuracy on all 5 seeds (interval "
            "excludes 0). For dropout and the LR scheduler the intervals include 0 and the seeds disagree in "
            "sign. These statements describe the synthetic-shape data only."
        )
    else:
        st.caption("A 95% interval that includes 0 means these runs do not show a clear difference. "
                   "Conclusions are written in the results report once all runs are complete.")
    st.subheader("Training and validation loss curves")
    _show_figure(ev, "loss_curves", "Training vs validation loss by variant (mean over seeds, band = min to max)")
    st.subheader("Training and validation accuracy curves")
    _show_figure(ev, "accuracy_curves", "Validation accuracy by variant; training accuracy where it was recorded")
    st.subheader("Confusion matrices")
    _show_figure(ev, "confusion_matrices", "Test confusion matrices by variant, summed over seeds")
    if not historical:
        _section("Per-class precision and recall",
                 lambda: evidence.load_per_class_metrics(ev.analysis_dir, ev.classes), lambda r: _render_per_class(ev, r))
        _section("Probability calibration (temperature scaling)",
                 lambda: evidence.load_calibration_summary(ev.analysis_dir), lambda r: _render_calibration(ev, r))
    st.divider()
    st.header("Grad-CAM gallery")
    _section("Correct and incorrect examples",
             lambda: evidence.load_gradcam_examples(ev.gradcam_dir, ev.gradcam_counts), lambda e: _render_gradcam(ev, e))


with compare_tab:
    st.header("Compare Models")
    fm = evidence.FASHION
    st.subheader(fm.title)
    if fm.available():
        prov = evidence.load_provenance(fm.sql_dir)
        st.info(
            f"**Saved experiment results** — group `{fm.group}`. Tables are read from CSV files saved in "
            f"`experiments/results/{fm.group}-sql/`; figures were generated from them. This tab displays files "
            "saved in the repository; it does not re-query Supabase or retrain."
        )
        if prov["is_supabase_sql"]:
            st.caption(f"SQL provenance: **{prov['source']}** (see PROVENANCE.md next to the CSVs).")
        else:
            st.warning(
                f"**SQL provenance: {prov['source'].rstrip('.')}.** These numbers are the real saved run records, but the CSVs "
                "were produced by the project's queries on a local copy, not by the Supabase SQL editor. They are "
                "provisional until the packaged SQL export has been run once in Supabase and ingested."
            )
        _render_evidence_set(fm, historical=False)
    else:
        st.warning(
            "**Final Fashion-MNIST results are not available yet.** The official 10-class comparison has not "
            "been completed, so no numbers, curves or Grad-CAM examples are shown for it here. Nothing below "
            "this section describes Fashion-MNIST."
        )

    st.divider()
    # A toggle (not an expander): the historical gallery itself uses expanders, which cannot be nested.
    hs = evidence.HISTORICAL
    if st.toggle(f"Show {hs.title} — earlier experiment, kept for transparency", value=False):
        st.warning(
            "**This is not Fashion-MNIST.** It is the earlier version of the project, which classified four "
            "generated shapes (horizontal bar, vertical bar, diagonal line, filled block) on 2,000 synthetic "
            f"images (group `{hs.group}`, 5 seeds × 6 variants = 30 runs). It was kept because the project "
            "starter shipped with that dataset, but the assignment asks for CIFAR-10 or Fashion-MNIST, so "
            "it is not the final evidence. These runs have no training-accuracy history or calibration."
        )
        gone = evidence.missing_figures(evidence.figures_for(hs))
        if gone:
            st.caption("Some historical figures are missing: " + ", ".join(f for f in gone if "per_class" not in f))
        _render_evidence_set(hs, historical=True)

# ---------------------------------------------------------------------------
# Model Card
# ---------------------------------------------------------------------------
with card_tab:
    try:
        st.markdown(evidence.load_model_card())
    except evidence.EvidenceError as exc:
        st.warning(f"The model card is unavailable. {exc}")
