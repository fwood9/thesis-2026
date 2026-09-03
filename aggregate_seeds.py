import os
import glob
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, FixedLocator, NullLocator
import statsmodels.formula.api as smf

BASE = "artifacts"
OUT = os.path.join("artifacts", "mean")
os.makedirs(OUT, exist_ok=True)

REFERENCE_RACE = "Caucasian"
CLASSES = {0: "NO", 1: ">30", 2: "<30"}
TEAL = {"NO": "#80CBC4", ">30": "#26A69A", "<30": "#00695C"}
OFFSETS = {"NO": 0.24, ">30": 0.0, "<30": -0.24}
ATTRS = ["race", "gender", "age", "race_age"]

# variant label - (suffix, predictions filename, plot title)
VARIANTS = {
    "baseline":   ("",              "predictions.joblib",               "Baseline"),
    "SMOTE":      ("_smote",        "predictions_smote.joblib",         "SMOTE model"),
    "Fair-SMOTE": ("_fair",         "predictions_fair.joblib",          "fair SMOTE model"),
    "DIR":        ("_dir",          "predictions_dir.joblib",           "DIR model"),
    "DIR_Many":   ("_DIR_Many",     "predictions_DIR_Many.joblib",     "DIR Many Model"),
    "CTAB-GAN":   ("_ctabgan_fair", "predictions_ctabgan_fair.joblib",  "fair CTAB-GAN model"),
}

seed_dirs = sorted(glob.glob(os.path.join(BASE, "seed_*")))
if not seed_dirs:
    raise SystemExit(f"no artifacts/seed_* folders found under {BASE}/")
print(f"found {len(seed_dirs)} seed folders: {[os.path.basename(d) for d in seed_dirs]}")

# shared, fixed inputs
split = joblib.load(os.path.join(BASE, "split.joblib"))
df_model = joblib.load(os.path.join(BASE, "df_model.joblib"))
X_test = split["X_test"]
race_test = df_model.loc[X_test.index, "race"].fillna("Missing").astype(str).values



# Helper functions

def per_seed_forest_or(y_pred_3way):
    """Fit the three one-vs-rest race logits, return tidy OR/CI per (class,race)."""
    d = pd.DataFrame({"race": race_test})
    rows = []
    for k, name in CLASSES.items():
        d["y"] = (y_pred_3way == k).astype(int)
        if d["y"].nunique() < 2:
            continue
        res = smf.logit(f'y ~ C(race, Treatment(reference="{REFERENCE_RACE}"))',
                        data=d).fit(disp=0)
        conf = res.conf_int()
        t = pd.DataFrame({"term": res.params.index,
                          "odds_ratio": np.exp(res.params.values),
                          "or_lo": np.exp(conf[0].values),
                          "or_hi": np.exp(conf[1].values)})
        t = t[t["term"].str.contains(r"\[T\.")].copy()
        t["race"] = t["term"].str.replace(r'.*\[T\.(.+)\]', r'\1', regex=True)
        t["class"] = name
        rows.append(t[["class", "race", "odds_ratio", "or_lo", "or_hi"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def mean_forest_plot(mean_or, title, path):
    race_counts = pd.Series(race_test).value_counts()
    races = [r for r in race_counts.index
             if r != REFERENCE_RACE and r in set(mean_or["race"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvspan(0.8, 1.25, color="grey", alpha=0.08, zorder=0)
    for yi in range(len(races)):
        ax.axhline(yi, color="0.9", lw=0.8, zorder=0)
    for name in CLASSES.values():
        sub = mean_or[mean_or["class"] == name].set_index("race").reindex(races)
        for yi, (_, row) in enumerate(sub.iterrows()):
            if np.isnan(row["odds_ratio"]):
                continue
            y = yi + OFFSETS[name]
            ax.plot([row["or_lo"], row["or_hi"]], [y, y], color=TEAL[name],
                    lw=2.4, solid_capstyle="round", zorder=2)
            ax.plot(row["odds_ratio"], y, "o", color=TEAL[name], ms=6.5,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ax.axvline(1.0, color="black", ls="--", lw=1, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 5.0)
    ax.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1.0, 2.0, 5.0]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks(range(len(races)))
    ax.set_yticklabels(races)
    ax.set_ylim(-0.6, len(races) - 0.4)
    ax.set_xlabel("Odds ratio vs Caucasian (log scale)")
    ax.set_title(f"Race allocation bias by predicted outcome ({title}, mean of "
                 f"{len(seed_dirs)} seeds)")
    ax.grid(axis="x", ls=":", alpha=0.4)
    handles = [Line2D([0], [0], color=TEAL[n], lw=2.4, marker="o", label=n)
               for n in CLASSES.values()]
    ax.legend(handles=handles, title="Predicted outcome", loc="upper left",
              bbox_to_anchor=(1.02, 1), frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def mean_over_seeds(paths, index_cols):
    """Read matching CSVs across seeds, return per-cell mean and SD."""
    frames = [pd.read_csv(p) for p in paths]
    cat = pd.concat(frames, ignore_index=True)
    num = cat.select_dtypes("number").columns.tolist()
    g = cat.groupby(index_cols, observed=True)
    mean = g[num].mean().reset_index()
    sd = g[num].std().reset_index()
    return mean, sd

# per-variant aggregation

for label, (suffix, pred_name, title) in VARIANTS.items():
    # Mean forest plot (mean OR point, averaged CI whiskers)
    or_frames = []
    for d in seed_dirs:
        pp = os.path.join(d, pred_name)
        if not os.path.exists(pp):
            continue
        y_pred_3way = np.asarray(joblib.load(pp)["y_pred_3way"])
        f = per_seed_forest_or(y_pred_3way)
        if not f.empty:
            or_frames.append(f)
    if or_frames:
        allf = pd.concat(or_frames, ignore_index=True)
        mean_or = (allf.groupby(["class", "race"], observed=True)
                   [["odds_ratio", "or_lo", "or_hi"]].mean().reset_index())
        sd_or = (allf.groupby(["class", "race"], observed=True)
                 ["odds_ratio"].std().reset_index().rename(columns={"odds_ratio": "or_sd_across_seeds"}))
        mean_or.merge(sd_or, on=["class", "race"]).round(4).to_csv(
            os.path.join(OUT, f"forest_or_mean{suffix}.csv"), index=False)
        mean_forest_plot(mean_or, title,
                         os.path.join(OUT, f"logit_forest_3way{suffix}_mean.png"))
        print(f"[{label:11s}] forest plot + OR table  ({len(or_frames)} seeds)")
    else:
        print(f"[{label:11s}] no predictions found - skipped")
        continue

    # mean confusion matrix (counts averaged; same fixed test set)
    cm_paths = [os.path.join(d, f"confusion_matrix_3way{suffix}.csv") for d in seed_dirs]
    cm_paths = [p for p in cm_paths if os.path.exists(p)]
    if cm_paths:
        cms = [pd.read_csv(p, index_col=0) for p in cm_paths]
        cm_mean = sum(cms) / len(cms)
        cm_mean.round(2).to_csv(os.path.join(OUT, f"confusion_matrix_3way{suffix}_mean.csv"))

    # mean fairness table (+ SD)
    for attr in ATTRS:
        fpaths = [os.path.join(d, f"fairness_{attr}{suffix}.csv") for d in seed_dirs]
        fpaths = [p for p in fpaths if os.path.exists(p)]
        if not fpaths:
            continue
        key = [attr, "class"] if attr != "race_age" else ["race_age", "class"]
        # fall back gracefully if the id columns differ
        sample = pd.read_csv(fpaths[0])
        key = [c for c in key if c in sample.columns]
        mean, sd = mean_over_seeds(fpaths, key)
        mean.round(4).to_csv(os.path.join(OUT, f"fairness_{attr}{suffix}_mean.csv"), index=False)
        sd.round(4).to_csv(os.path.join(OUT, f"fairness_{attr}{suffix}_sd.csv"), index=False)

print("\nsaved mean artifacts to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)



paths = sorted(glob.glob("artifacts/seed_*/overall_importance.csv"))
print(f"found {len(paths)} seed files")

# stack every seed's per-feature importance_mean, keyed by feature
wide = pd.concat(
    [pd.read_csv(p).set_index("feature")["importance_mean"].rename(os.path.basename(os.path.dirname(p)))
     for p in paths],
    axis=1,
)

agg = pd.DataFrame({
    "importance_mean_across_seeds": wide.mean(axis=1),   # mean of the per-seed means
    "importance_std_across_seeds":  wide.std(axis=1, ddof=1),  # SD across seeds
    "n_seeds": wide.notna().sum(axis=1),
}).sort_values("importance_mean_across_seeds", ascending=False)

agg.round(6).to_csv("artifacts/mean/overall_importance_across_seeds.csv")
print(agg.round(5).to_string())