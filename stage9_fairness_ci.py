import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator, NullLocator
from matplotlib.lines import Line2D
import statsmodels.formula.api as smf

SEED = int(os.environ.get("SEED", 10))
BASE = "artifacts"                               # fixed stage 1-2 inputs (never reseeded)
ART = os.path.join("artifacts", f"seed_{SEED}")  # per-seed outputs
os.makedirs(ART, exist_ok=True)

df_model = joblib.load(os.path.join(BASE, "df_model.joblib"))
split = joblib.load(os.path.join(BASE, "split.joblib"))
predictions = joblib.load(os.path.join(ART, "predictions.joblib"))

X_test = split["X_test"]
y_pred_3way = predictions["y_pred_3way"]

REFERENCE_RACE = "Caucasian"

d = pd.DataFrame({
    "race": df_model.loc[X_test.index, "race"].fillna("Missing").astype(str).values,
    "flagged_ge1": (y_pred_3way >= 1).astype(int),  # predicted "=>30" (cutoff t1)
    "flagged_lt30": (y_pred_3way == 2).astype(int),  # predicted "<30"    (cutoff t2)
})


# Cumulative binary logits

def tidy_logit(res):
    conf = res.conf_int()
    out = pd.DataFrame({
        "term": res.params.index,
        "coef": res.params.values,
        "odds_ratio": np.exp(res.params.values),
        "or_lo": np.exp(conf[0].values),
        "or_hi": np.exp(conf[1].values),
        "p_value": res.pvalues.values,
    })
    out["term"] = out["term"].str.replace(
        r'C\(race, Treatment\(reference="' + REFERENCE_RACE + r'"\)\)\[T\.(.+)\]',
        r'race[\1]', regex=True)
    return out


def fit_and_report(outcome):
    formula = f'{outcome} ~ C(race, Treatment(reference="{REFERENCE_RACE}"))'
    res = smf.logit(formula, data=d).fit(disp=0)
    table = tidy_logit(res)
    print(f"\n===== {outcome} ~ race (unadjusted, reference race: {REFERENCE_RACE}) =====")
    print(table.round(4).to_string(index=False))
    print("\n--- race terms only ---")
    print(table[table["term"].str.startswith("race[")].round(4).to_string(index=False))
    table.to_csv(os.path.join(ART, f"logit_{outcome}.csv"), index=False)
    return table


fit_and_report("flagged_ge1")
fit_and_report("flagged_lt30")


# Map the forest plots
classes = {0: "NO", 1: ">30", 2: "<30"}
TEAL    = {"NO": "#80CBC4", ">30": "#26A69A", "<30": "#00695C"}   # 3 different shades of teal
offsets = {"NO": 0.24, ">30": 0.0, "<30": -0.24}

results = {}
for k, name in classes.items():
    d["y"] = (y_pred_3way == k).astype(int)
    res = smf.logit(f'y ~ C(race, Treatment(reference="{REFERENCE_RACE}"))',
                    data=d).fit(disp=0)
    conf = res.conf_int()
    t = pd.DataFrame({
        "term": res.params.index,
        "odds_ratio": np.exp(res.params.values),
        "or_lo": np.exp(conf[0].values),
        "or_hi": np.exp(conf[1].values),
    })
    t = t[t["term"].str.contains(r"\[T\.")].copy()
    t["label"] = t["term"].str.replace(r'.*\[T\.(.+)\]', r'\1', regex=True)
    results[name] = t.set_index("label")

# fixed race order (largest group first) so every variant's plot matches
race_counts = d["race"].value_counts()
races = [r for r in race_counts.index if r != REFERENCE_RACE and r in results["<30"].index]

fig, ax = plt.subplots(figsize=(8, 5))
ax.axvspan(0.8, 1.25, color="grey", alpha=0.08, zorder=0)          # negligible band
for yi in range(len(races)):
    ax.axhline(yi, color="0.9", lw=0.8, zorder=0)                  # row guides

for name in classes.values():
    t = results[name].reindex(races)
    for yi, (_, row) in enumerate(t.iterrows()):
        y = yi + offsets[name]
        ax.plot([row["or_lo"], row["or_hi"]], [y, y], color=TEAL[name],
                lw=2.4, solid_capstyle="round", zorder=2)
        ax.plot(row["odds_ratio"], y, "o", color=TEAL[name], ms=6.5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)

ax.axvline(1.0, color="black", ls="--", lw=1, zorder=1)
ax.set_xscale("log")
ax.set_xlim(0.2, 5.0)                                          # captures every point + CI, symmetric about 1
ax.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1.0, 2.0, 5.0]))
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_minor_locator(NullLocator())
ax.set_yticks(range(len(races)))
ax.set_yticklabels(races)
ax.set_ylim(-0.6, len(races) - 0.4)
ax.set_xlabel("Odds ratio vs Caucasian (log scale)")
ax.set_title("Race allocation bias by predicted outcome (Baseline)")
ax.grid(axis="x", ls=":", alpha=0.4)

handles = [Line2D([0], [0], color=TEAL[n], lw=2.4, marker="o", label=n)
           for n in classes.values()]
ax.legend(handles=handles, title="Predicted outcome", loc="upper left",
          bbox_to_anchor=(1.02, 1), frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(ART, "logit_forest_3way.png"), dpi=200, bbox_inches="tight")


