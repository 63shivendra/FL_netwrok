"""
Privacy-Preserving Federated Learning for Hospitals: project showcase.

Run locally:  streamlit run streamlit_app.py
All numbers come from app_data/*.json, produced by experiments/run_experiment.py
(a faithful re-run of federated/client.py + federated/server.py).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Federated Learning for Hospitals", page_icon="🏥", layout="wide")

DATA = Path(__file__).parent / "app_data"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8f98"
CLASS_COLORS = ["#6250d6", "#eda100", "#1baf7a"]


@st.cache_data
def load():
    j = lambda n: json.loads((DATA / n).read_text())
    return j("profile.json"), j("fl_results.json"), j("baselines.json"), j("v2_results.json")


P, FL, B, V2 = load()
V2L = V2["rounds"][-1]
V2B = V2["baselines"]
bal = lambda mm: mm.get("bal_acc", float(np.mean([mm["per_class"][c]["recall"] for c in "012"])))
R = FL["rounds"]
LAST = R[-1]
SPL = FL["splits"]
MAJ = B["majority"]["acc"]
pct = lambda v, d=1: f"{v * 100:.{d}f}%"


def style(fig, h=320, yfmt=None):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified")
    fig.update_xaxes(showgrid=False)
    if yfmt:
        fig.update_yaxes(tickformat=yfmt)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ════════════════════════ HEADER + MAIN OBJECTIVE ════════════════════════
st.title("Privacy-Preserving Federated Learning for Hospitals")
st.markdown("##### Distributed machine learning across hospitals, where patient records never leave the hospital.")

st.info(
    "**Main objective.** Hospitals want better disease-prediction models, but privacy laws (HIPAA, GDPR) "
    "stop them from pooling patient records. This project shows that hospitals can **train together without "
    "sharing any raw data**. Each hospital trains on its own patients, and only model outputs travel to a "
    "central server. This approach is called **federated learning**, a form of privacy-preserving, distributed machine learning."
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**🔒 Privacy**  \nPatient rows stay inside each hospital. Only predicted probabilities are shared.")
with c2:
    st.markdown("**🌐 Distributed**  \nEach hospital trains locally, and each can use a different model family "
                "(here a neural network and an SVM).")
with c3:
    st.markdown("**🤝 Collaborative**  \nA server averages the hospitals' predictions and sends the consensus back, "
                "so each model learns from the other.")

m = st.columns(5)
m[0].metric("Patients (synthetic)", f"{P['rows']:,}")
m[1].metric("Hospitals", "2", help="Hospital 1: PyTorch MLP · Hospital 2: scikit-learn SVM")
m[2].metric("Raw records shared", "0")
m[3].metric("Macro-F1 · MLP (fixed v2)", f"{V2L['mlp']['macro_f1']:.3f}", f"{V2L['mlp']['macro_f1'] - LAST['mlp']['macro_f1']:+.3f} vs original", help="Macro-F1 weights all three prognosis classes equally")
m[4].metric("Balanced accuracy · MLP (v2)", pct(V2L["mlp"]["bal_acc"]), f"{(V2L['mlp']['bal_acc'] - bal(LAST['mlp'])) * 100:+.1f} pts vs original")

st.divider()
tabs = st.tabs(["How it works", "Data", "Results", "Fixes & improved results", "Try the simulator"])

# ════════════════════════ HOW IT WORKS ════════════════════════
with tabs[0]:
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("One federated round")
        st.graphviz_chart(f"""
digraph G {{
  rankdir=TB; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#9aa3ad", fillcolor="#f4f6f8"];
  edge [fontname="Helvetica", fontsize=9, color="#7b8794"];
  S  [label="Aggregation server\\naverages probabilities", fillcolor="#dcecee"];
  H1 [label="Hospital 1 · MLP (PyTorch)\\n{SPL['hospital1']:,} private patients"];
  H2 [label="Hospital 2 · SVM (scikit-learn)\\n{SPL['hospital2']:,} private patients"];
  RS [label="Shared reference set\\n{SPL['comm']:,} rows, no labels used"];
  RS -> H1 [label=" score"]; RS -> H2 [label=" score"];
  H1 -> S [label=" probabilities"]; H2 -> S [label=" probabilities"];
  S -> H1 [label=" consensus", style=dashed]; S -> H2 [label=" consensus", style=dashed];
}}""")
    with right:
        st.subheader("The six steps")
        st.markdown(f"""
1. **Train locally.** Each hospital trains on its own private patients.
2. **Score the reference set.** Both models predict class probabilities for the same {SPL['comm']:,} unlabeled rows.
3. **Upload probabilities only.** That is a {SPL['comm']:,} × 3 matrix per hospital. No weights, no gradients, no patient data.
4. **Average.** The server computes p̄ = (p_MLP + p_SVM) / 2, a soft "consensus" label.
5. **Fine-tune on the consensus.** The MLP minimizes soft-label cross-entropy. The SVM retrains on the pseudo-labels, weighted by confidence.
6. **Evaluate** on {SPL['test']:,} held-out test patients, then repeat for 5 rounds.
""")
        st.success("**Why not FedAvg?** FedAvg averages model *weights*, so every hospital needs the same "
                   "architecture. Sharing *predictions* lets a neural network and an SVM collaborate. "
                   "This technique is called federated distillation.")

    st.subheader("Explore a round")
    r = st.select_slider("Round", options=[x["round"] for x in R], value=R[-1]["round"])
    rd = R[r - 1]
    k = st.columns(4)
    k[0].metric("MLP accuracy", pct(rd["acc_mlp"], 2), f"{(rd['acc_mlp'] - rd['pre_distill']['mlp']['acc']) * 100:+.2f} pts from fine-tune")
    k[1].metric("SVM accuracy", pct(rd["acc_svm"], 2), f"{(rd['acc_svm'] - rd['pre_distill']['svm']['acc']) * 100:+.2f} pts from fine-tune")
    k[2].metric("Model agreement", pct(rd["agreement"], 1), help="Share of reference rows where both models predict the same class (before fine-tuning)")
    k[3].metric("SVM best params", f"C={rd['svm_params']['C']}, γ={rd['svm_params']['gamma']}")

    with st.expander("See the key code (client.py)"):
        a, b = st.columns(2)
        a.markdown("**Hospital 1: soft-label fine-tune**")
        a.code("""outputs = model(comm_data)
loss = -(averaged_probs * F.log_softmax(outputs, dim=1)).sum(dim=1).mean()
loss.backward(); optimizer.step()""", language="python")
        b.markdown("**Hospital 2: confidence-weighted pseudo-labels**")
        b.code("""labels = np.argmax(averaged_probs, axis=1)
weights = np.max(averaged_probs, axis=1)
grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3)
grid.fit(comm_data, labels, sample_weight=weights)""", language="python")

# ════════════════════════ DATA ════════════════════════
with tabs[1]:
    st.subheader("From FHIR records to a training table")
    st.markdown(f"""
| Stage | What happens | File |
|---|---|---|
| Parse | Synthea FHIR JSON bundles → one row per patient | `dataprocees.py` |
| Clean | Age from birth date; missing labs filled with the **age-group median** | `datahandlingmissingvalues.py` |
| Final table | **{P['rows']:,} patients × 44 numeric features**, 3-class `prognosis` label, {P['missing_total']} missing values | `finaldataset.csv` |
| Split | Stratified: 30% Hospital 1 · 30% Hospital 2 · 20% shared reference · 20% test | `server.py` |
""")
    a, b = st.columns(2)
    with a:
        names = ["Hospital 1", "Hospital 2", "Reference set", "Test set"]
        vals = [SPL["hospital1"], SPL["hospital2"], SPL["comm"], SPL["test"]]
        fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=[BLUE, ORANGE, AQUA, GRAY],
                               text=[f"{v:,}" for v in vals], textposition="outside"))
        fig.update_layout(title="Who holds which rows")
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 280))
    with b:
        cc = P["class_counts"]
        tot = sum(cc.values())
        fig = go.Figure(go.Bar(x=[f"Class {k}" for k in cc], y=list(cc.values()), marker_color=CLASS_COLORS,
                               text=[pct(v / tot) for v in cc.values()], textposition="outside"))
        fig.update_layout(title="Class balance: heavily skewed")
        show(style(fig, 280))
    st.warning(f"**Class imbalance matters.** {pct(cc['1'] / tot)} of patients are class 1, so a model that always "
               f"predicts class 1 already gets **{pct(MAJ)} accuracy**. That's why the results also report macro-F1.")

    with st.expander("Which features separate the classes most?"):
        rows = [{"Feature": f["name"], "ANOVA F": round(f["F"]),
                 **{f"Class {c} (z)": round(f["class_means_z"][c], 2) for c in "012"}} for f in P["top_features"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("z = class mean in standard deviations from the overall mean.")
    with st.expander("Age distribution"):
        e, c = P["age_hist"]["edges"], P["age_hist"]["counts"]
        fig = go.Figure(go.Bar(x=[f"{e[i]}–{e[i + 1]}" for i in range(len(c))], y=c, marker_color=BLUE))
        show(style(fig, 260))

# ════════════════════════ RESULTS ════════════════════════
with tabs[2]:
    st.subheader("Results from a full re-run of the original code")
    st.caption("client.py + server.py unchanged · seed 42 · 5 rounds · 10 epochs per stage · scored on 13,236 held-out patients")

    t1, t2, t3 = st.columns(3)
    t1.success(f"**Federation helped the SVM.** Alone it predicts only the majority class (macro-F1 "
               f"{B['local_svm_h2']['macro_f1']:.2f}). Federated, it reaches **{LAST['svm']['macro_f1']:.2f}**.")
    t2.info(f"**The MLP's ROC-AUC improved** from {R[0]['mlp']['auc_ovr']:.3f} to **{LAST['mlp']['auc_ovr']:.3f}**, "
            f"close to the pooled-data model ({B['central_mlp']['auc_ovr']:.3f}).")
    t3.warning(f"**Accuracy is near the {pct(MAJ)} baseline**, and minority-class recall fell over the rounds. "
               "Class imbalance is the main thing to fix next.")

    rounds = [x["round"] for x in R]
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        fig.add_scatter(x=rounds, y=[x["acc_mlp"] for x in R], name="Hospital 1 · MLP", line=dict(color=BLUE, width=2), mode="lines+markers")
        fig.add_scatter(x=rounds, y=[x["acc_svm"] for x in R], name="Hospital 2 · SVM", line=dict(color=ORANGE, width=2), mode="lines+markers")
        fig.add_hline(y=MAJ, line_dash="dash", line_color=GRAY, annotation_text="always predict class 1", annotation_position="bottom right")
        fig.update_layout(title="Test accuracy by round")
        fig.update_xaxes(title="Round", dtick=1)
        show(style(fig, 330, ".1%"))
    with b:
        fig = go.Figure()
        fig.add_scatter(x=rounds, y=[x["mlp"]["macro_f1"] for x in R], name="Hospital 1 · MLP", line=dict(color=BLUE, width=2), mode="lines+markers")
        fig.add_scatter(x=rounds, y=[x["svm"]["macro_f1"] for x in R], name="Hospital 2 · SVM", line=dict(color=ORANGE, width=2), mode="lines+markers")
        fig.update_layout(title="Macro-F1 by round (all classes weighted equally)")
        fig.update_xaxes(title="Round", dtick=1)
        show(style(fig, 330, ".2f"))

    st.markdown("**Is federation worth it? Every model scored on the same test set**")

    def row(name, mm):
        return {"Model": name, "Accuracy": mm["acc"], "Macro-F1": mm["macro_f1"],
                "Recall · class 0": mm["per_class"]["0"]["recall"], "Recall · class 2": mm["per_class"]["2"]["recall"],
                "ROC-AUC": mm.get("auc_ovr")}

    tbl = pd.DataFrame([
        row("Always predict class 1", B["majority"]),
        row("Hospital 2 SVM, trained alone", B["local_svm_h2"]),
        row("Hospital 1 MLP, trained alone", B["local_mlp_h1"]),
        row("⭐ Federated SVM (round 5)", LAST["svm"]),
        row("⭐ Federated MLP (round 5)", LAST["mlp"]),
        row("All data pooled (upper bound)", B["central_mlp"]),
        row("All data pooled, class-weighted", B["central_mlp_weighted"]),
    ])
    st.dataframe(tbl.style.format({c: "{:.3f}" for c in tbl.columns if c != "Model"}, na_rep="–"),
                 hide_index=True, width="stretch")

    with st.expander("Training loss and model agreement"):
        a, b = st.columns(2)
        with a:
            fig = go.Figure()
            fig.add_scatter(y=FL["train_losses"], name="Local training (cross-entropy)", line=dict(color=BLUE, width=2))
            fig.add_scatter(y=FL["ft_losses"], name="Fine-tune (soft labels)", line=dict(color=AQUA, width=2))
            fig.update_layout(title="Hospital 1 loss (10 epochs × 5 rounds)")
            fig.update_xaxes(title="Epoch")
            show(style(fig, 300))
        with b:
            fig = go.Figure(go.Scatter(x=rounds, y=[x["agreement"] for x in R], line=dict(color=BLUE, width=2), mode="lines+markers"))
            fig.update_layout(title="MLP–SVM agreement on the reference set")
            fig.update_xaxes(dtick=1, title="Round")
            show(style(fig, 300, ".1%"))
    with st.expander("Confusion matrices (final round)"):
        a, b = st.columns(2)
        for col, key, name in [(a, "mlp", "Hospital 1 · MLP"), (b, "svm", "Hospital 2 · SVM")]:
            cm = np.array(LAST[key]["cm"])
            share = cm / cm.sum(1, keepdims=True)
            fig = go.Figure(go.Heatmap(z=share, x=["Pred 0", "Pred 1", "Pred 2"], y=["True 0", "True 1", "True 2"],
                                       colorscale="Blues", showscale=False,
                                       text=[[f"{cm[i][j]:,}<br>{share[i][j]:.0%}" for j in range(3)] for i in range(3)],
                                       texttemplate="%{text}"))
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(title=name)
            col.plotly_chart(style(fig, 300), width="stretch", config={"displayModeBar": False})

# ════════════════════════ SIMULATOR ════════════════════════
H_UNITS, D, C = 16, 44, 3


@st.cache_data
def sim_data():
    d = json.loads((DATA / "sim_sample.json").read_text())
    return np.array(d["X"]), np.array(d["y"]), np.array(d["Xt"]), np.array(d["yt"])


def new_model(rng):
    return {"W1": rng.normal(0, np.sqrt(2 / D), (D, H_UNITS)), "b1": np.zeros(H_UNITS),
            "W2": rng.normal(0, np.sqrt(2 / H_UNITS), (H_UNITS, C)), "b2": np.zeros(C)}


def copy(m):
    return {k: v.copy() for k, v in m.items()}


def forward(m, X):
    h = np.maximum(0, X @ m["W1"] + m["b1"])
    z = h @ m["W2"] + m["b2"]
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return h, e / e.sum(1, keepdims=True)


def train(m, X, T, cw, epochs, rng, lr=0.05, bs=32):
    n = len(X)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            i = perm[s:s + bs]
            x, t = X[i], T[i]
            h, p = forward(m, x)
            w = (t @ cw)[:, None] if cw is not None else 1.0
            dz = (p - t) * w / len(i)
            dh = (dz @ m["W2"].T) * (h > 0)
            m["W2"] -= lr * (h.T @ dz + 1e-4 * m["W2"]); m["b2"] -= lr * dz.sum(0)
            m["W1"] -= lr * (x.T @ dh + 1e-4 * m["W1"]); m["b1"] -= lr * dh.sum(0)


def score(models, X, y):
    accs, f1s = [], []
    for m in models:
        pred = forward(m, X)[1].argmax(1)
        accs.append((pred == y).mean())
        f = []
        for c in range(C):
            tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
            f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        f1s.append(np.mean(f))
    return float(np.mean(accs)), float(np.mean(f1s))


@st.cache_data(show_spinner="Training hospitals…")
def simulate(K, alpha, rounds, epochs, noise, drop, weighted, seed):
    X, y, Xt, yt = sim_data()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    ref, pool = idx[:600], idx[600:]
    parts = [[] for _ in range(K)]
    for c in range(C):
        ci = pool[y[pool] == c]
        props = rng.dirichlet([alpha] * K)
        cuts = (np.cumsum(props)[:-1] * len(ci)).astype(int)
        for k, chunk in enumerate(np.split(ci, cuts)):
            parts[k].extend(chunk.tolist())
    parts = [np.array(p) for p in parts]
    for k in range(K):  # every hospital gets at least a few patients
        if len(parts[k]) < 10:
            big = int(np.argmax([len(p) for p in parts]))
            parts[k], parts[big] = np.concatenate([parts[k], parts[big][:30]]), parts[big][30:]
    onehot = np.eye(C)
    Xp = [X[p] for p in parts]; Tp = [onehot[y[p]] for p in parts]
    Xref = X[ref]
    allp = np.concatenate(parts)
    cwf = (lambda yy: len(yy) / (C * np.maximum(1, np.bincount(yy, minlength=C)))) if weighted else (lambda yy: None)
    cws = [cwf(y[p]) for p in parts]
    init = new_model(np.random.default_rng(seed + 99))
    local = [copy(init) for _ in range(K)]
    distill = [copy(init) for _ in range(K)]
    glob, pooled = copy(init), copy(init)
    hist = []
    for t in range(rounds):
        active = rng.random(K) >= drop
        if not active.any():
            active[rng.integers(K)] = True
        for k in range(K):
            train(local[k], Xp[k], Tp[k], cws[k], epochs, rng)
        # FedAvg: weighted average of weights
        ups, ns = [], []
        for k in np.where(active)[0]:
            mk = copy(glob); train(mk, Xp[k], Tp[k], cws[k], epochs, rng)
            if noise:
                for key in mk: mk[key] += rng.normal(0, noise * 0.05, mk[key].shape)
            ups.append(mk); ns.append(len(parts[k]))
        w = np.array(ns) / sum(ns)
        glob = {key: sum(wi * u[key] for wi, u in zip(w, ups)) for key in glob}
        # Probability averaging (this project)
        probs = []
        for k in range(K):
            train(distill[k], Xp[k], Tp[k], cws[k], epochs, rng)
            if active[k]:
                p = forward(distill[k], Xref)[1]
                if noise:
                    p = np.clip(p + rng.normal(0, noise * 0.25, p.shape), 1e-4, None); p /= p.sum(1, keepdims=True)
                probs.append(p)
        avg = np.mean(probs, axis=0)
        for k in range(K):
            train(distill[k], Xref, avg, None, 1, rng)
        train(pooled, X[allp], onehot[y[allp]], cwf(y[allp]), epochs, rng)
        for name, models in [("Probability averaging (this project)", distill), ("FedAvg", [glob]),
                             ("All data pooled", [pooled]), ("Each hospital alone", local)]:
            a, f = score(models, Xt, yt)
            hist.append({"Round": t + 1, "Strategy": name, "Accuracy": a, "Macro-F1": f})
    comp = [np.bincount(y[p], minlength=C).tolist() for p in parts]
    return pd.DataFrame(hist), comp, float(np.bincount(yt).max() / len(yt))


# ════════════════════════ FIXES (v2) ════════════════════════
with tabs[3]:
    st.subheader("Bugs fixed: original vs. v2")
    st.caption("Same data, same 30/30/20/20 split, same 13,236 test patients. The v2 settings were chosen on a validation split "
               "carved from training data, and the test set was used once. Code: federated_v2/ · experiments/run_v2.py")

    o1, o2, n1, n2 = LAST["mlp"], LAST["svm"], V2L["mlp"], V2L["svm"]
    k = st.columns(4)
    k[0].metric("MLP macro-F1", f"{n1['macro_f1']:.3f}", f"{n1['macro_f1'] - o1['macro_f1']:+.3f} vs original")
    k[1].metric("SVM macro-F1", f"{n2['macro_f1']:.3f}", f"{n2['macro_f1'] - o2['macro_f1']:+.3f} vs original")
    k[2].metric("MLP balanced accuracy", pct(n1["bal_acc"]), f"{(n1['bal_acc'] - bal(o1)) * 100:+.1f} pts")
    k[3].metric("MLP accuracy", pct(n1["acc"]), f"{(n1['acc'] - o1['acc']) * 100:+.1f} pts (expected)", delta_color="off")

    st.info(f"**Why plain accuracy went down.** The original models scored about {pct(o1['acc'])} mostly by predicting class 1 "
            f"(only {pct(o1['per_class']['0']['recall'])} of class-0 patients were caught). v2 now catches "
            f"**{pct(n1['per_class']['0']['recall'])}** of class 0 and **{pct(n1['per_class']['2']['recall'])}** of class 2, "
            "at the cost of some false alarms on class 1. For a medical model, finding the rare cases matters more.")

    rounds_v2 = [x["round"] for x in V2["rounds"]]
    fig = go.Figure()
    fig.add_scatter(x=rounds, y=[x["mlp"]["macro_f1"] for x in R], name="MLP · original", line=dict(color=BLUE, width=2, dash="dot"))
    fig.add_scatter(x=rounds, y=[x["svm"]["macro_f1"] for x in R], name="SVM · original", line=dict(color=ORANGE, width=2, dash="dot"))
    fig.add_scatter(x=rounds_v2, y=[x["mlp"]["macro_f1"] for x in V2["rounds"]], name="MLP · v2", line=dict(color=BLUE, width=2), mode="lines+markers")
    fig.add_scatter(x=rounds_v2, y=[x["svm"]["macro_f1"] for x in V2["rounds"]], name="SVM · v2", line=dict(color=ORANGE, width=2), mode="lines+markers")
    fig.update_layout(title="Macro-F1 by round: original (dotted) vs. v2 (solid)")
    fig.update_xaxes(title="Round", dtick=1)
    show(style(fig, 340, ".2f"))

    def r2(name, mm):
        return {"Model": name, "Accuracy": mm["acc"], "Macro-F1": mm["macro_f1"], "Balanced acc": bal(mm),
                "Recall · class 0": mm["per_class"]["0"]["recall"], "Recall · class 2": mm["per_class"]["2"]["recall"],
                "ROC-AUC": mm.get("auc_ovr")}

    cmp = pd.DataFrame([
        r2("Original · federated MLP", o1), r2("Original · federated SVM", o2),
        r2("⭐ v2 · federated MLP", n1), r2("⭐ v2 · federated SVM", n2),
        r2("v2 · MLP trained alone", V2B["local_mlp_h1"]), r2("v2 · SVM trained alone", V2B["local_svm_h2"]),
        r2("v2 · all data pooled (upper bound)", V2B["central_mlp"]),
    ])
    st.dataframe(cmp.style.format({c: "{:.3f}" for c in cmp.columns if c != "Model"}, na_rep="–"), hide_index=True, width="stretch")

    st.markdown("**What was fixed**")
    st.markdown("""
| # | Bug | Fix in v2 |
|---|---|---|
| 1 | Class imbalance ignored, so models mostly predicted class 1 | Loss weighted by √(inverse class frequency), from each hospital's own labels |
| 2 | SVM threw away its own training every round | Fine-tune retrains on **local labels + consensus pseudo-labels** |
| 3 | SVM "probabilities" were a softmax over raw scores | Platt-calibrated on 10% of the hospital's own data |
| 4 | MLP fine-tune drifted toward the majority-class consensus | Mixed loss: 50% soft consensus + 50% local labels, mini-batched |
| 5 | Scaler was fit on both hospitals' pooled rows | **Federated scaler**: hospitals share only count, sum, sum of squares |
| 6 | Agreement measured before fine-tune; "improvement" vs round 1 | Agreement measured after fine-tune; compared against hospitals training alone |
""")
    with st.expander("Confusion matrices (v2, final round)"):
        a, b = st.columns(2)
        for col, key, name in [(a, "mlp", "Hospital 1 · MLP (v2)"), (b, "svm", "Hospital 2 · SVM (v2)")]:
            cm = np.array(V2L[key]["cm"])
            share = cm / cm.sum(1, keepdims=True)
            fig = go.Figure(go.Heatmap(z=share, x=["Pred 0", "Pred 1", "Pred 2"], y=["True 0", "True 1", "True 2"],
                                       colorscale="Blues", showscale=False,
                                       text=[[f"{cm[i][j]:,}<br>{share[i][j]:.0%}" for j in range(3)] for i in range(3)],
                                       texttemplate="%{text}"))
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(title=name)
            col.plotly_chart(style(fig, 300), width="stretch", config={"displayModeBar": False})
    with st.expander("What I tried that did not help"):
        st.markdown(f"- **Missing-value indicator columns** for the 25 mostly-imputed features: {V2['choices']['missing_indicators']}.\n"
                    "- **Full inverse-frequency weights:** higher minority recall, but validation macro-F1 dropped to 0.54 (√ weights gave 0.61).")


with tabs[4]:
    st.subheader("Federation simulator")
    st.markdown("This trains real neural networks on a 3,000-patient sample of the project data (1,000 test patients, IDs removed). "
                "Change the settings and compare **this project's probability averaging** with **FedAvg**, "
                "**each hospital training alone**, and **pooling all the data**, which is the non-private upper bound.")
    c = st.columns(4)
    K = c[0].slider("Hospitals", 2, 6, 3)
    het = c[1].select_slider("Data heterogeneity", ["IID", "Low", "Medium", "High", "Extreme"], value="Medium",
                             help="How unevenly the classes are spread across hospitals (Dirichlet label skew)")
    rounds = c[2].slider("Rounds", 5, 30, 15)
    epochs = c[3].slider("Local epochs / round", 1, 5, 1)
    c = st.columns(4)
    noise = c[0].slider("Noise on shared values", 0.0, 1.0, 0.0, 0.1,
                        help="Gaussian noise on whatever leaves a hospital: a simple stand-in for differential privacy (not a formal ε guarantee)")
    drop = c[1].slider("Client dropout per round", 0, 50, 0, 10, format="%d%%")
    weighted = c[2].toggle("Class-weighted loss", help="Counter the class imbalance")
    seed = c[3].number_input("Random seed", 0, 9999, 11)

    alpha = {"IID": 1000, "Low": 3, "Medium": 1, "High": 0.3, "Extreme": 0.1}[het]
    df, comp, maj = simulate(K, alpha, rounds, epochs, noise, drop / 100, weighted, int(seed))

    colors = {"Probability averaging (this project)": BLUE, "FedAvg": ORANGE, "All data pooled": AQUA, "Each hospital alone": GRAY}
    a, b = st.columns(2)
    for col, metric, fmt in [(a, "Accuracy", ".1%"), (b, "Macro-F1", ".2f")]:
        fig = go.Figure()
        for name, color in colors.items():
            d = df[df.Strategy == name]
            fig.add_scatter(x=d.Round, y=d[metric], name=name, line=dict(color=color, width=2))
        if metric == "Accuracy":
            fig.add_hline(y=maj, line_dash="dash", line_color=GRAY, annotation_text="always class 1", annotation_position="bottom right")
        fig.update_layout(title=f"{metric} by round")
        fig.update_xaxes(title="Round")
        col.plotly_chart(style(fig, 360, fmt), width="stretch", config={"displayModeBar": False})

    final = df[df.Round == rounds].set_index("Strategy")[["Accuracy", "Macro-F1"]]
    a, b = st.columns([1, 1])
    a.markdown("**Final round**")
    a.dataframe(final.style.format("{:.3f}"), width="stretch")
    b.markdown("**Each hospital's patients (class 0 / 1 / 2)**")
    b.dataframe(pd.DataFrame(comp, columns=["Class 0", "Class 1", "Class 2"],
                             index=[f"Hospital {i + 1}" for i in range(K)]), width="stretch")
    st.caption("Try this: set heterogeneity to *Extreme*. Hospitals training alone fall apart, and federation recovers much of the gap. "
               "Then raise the noise to see the privacy-versus-accuracy trade-off.")

st.divider()
st.caption("Data is synthetic (Synthea), with no real patients. Code: federated/client.py and federated/server.py. "
           "Results: experiments/run_experiment.py → app_data/. "
           "[GitHub repository](https://github.com/63shivendra/FL_netwrok)")
