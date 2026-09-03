import os
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

ART = "artifacts"
OUT = os.path.join(ART, "proxy_audit_unadjusted")
os.makedirs(OUT, exist_ok=True)

REF = "Caucasian"
FOCUS = "AfricanAmerican"
MIN_N = 100

df = joblib.load(os.path.join(ART, "df_model.joblib")).copy()
df["readmit_any"] = (df["readmit_class"] >= 1).astype(int)
df["race"] = df["race"].fillna("Missing").astype(str)
df["disp"] = df["discharged_to_another_facility"].astype(int)
df["number_inpatient"] = pd.to_numeric(df["number_inpatient"], errors="coerce")

age_bins = [0, 20, 40, 50, 60, 70, 80, 100]
age_labels = ["<20", "20-40", "40-50", "50-60", "60-70", "70-80", "80+"]
df["age_band"] = pd.cut(df["age_numeric"], bins=age_bins, labels=age_labels,
                        right=False, include_lowest=True)

race_order = [r for r, n in df["race"].value_counts().items() if n >= MIN_N]
palette = {r: c for r, c in zip(race_order, plt.cm.tab10(np.linspace(0, 1, len(race_order))))}
palette[FOCUS] = "#c0392b"
palette[REF] = "#2471a3"


def prop_ci(k, n):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    se = np.sqrt(p * (1 - p) / n)
    return p, max(0, p - 1.96 * se), min(1, p + 1.96 * se)


def mean_ci(x):
    x = x.dropna()
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x))
    return m, m - 1.96 * se, m + 1.96 * se


summary = []



# Rate by race
rows = []
for r in race_order:
    g = df[df.race == r]
    p, lo, hi = prop_ci(g["disp"].sum(), len(g))
    rows.append(dict(race=r, rate=p, lo=lo, hi=hi, n=len(g)))
A = pd.DataFrame(rows).sort_values("rate", ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(A.race, A.rate, color=[palette[r] for r in A.race],
       yerr=[A.rate - A.lo, A.hi - A.rate], capsize=5)
ax.axhline(df["disp"].mean(), color="darkorange", ls="--", lw=2, label=f"Overall = {df['disp'].mean():.1%}")
for i, row in enumerate(A.itertuples()):
    ax.text(i, row.rate + 0.005, f"{row.rate:.1%}\nn={row.n:,}", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("P(discharged to another facility)")
ax.set_title("A1. Discharge-to-facility rate by race")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "A1_disp_rate_by_race.png"), dpi=150, bbox_inches="tight")
plt.close()

# Rate by race x age (descriptive context)
fig, ax = plt.subplots(figsize=(10, 5.5))
for r in race_order:
    ys = []
    for b in age_labels:
        s = df[(df.race == r) & (df.age_band == b)]
        ys.append(s["disp"].mean() if len(s) >= 50 else np.nan)
    ax.plot(age_labels, ys, "-o", color=palette[r], lw=2, label=r)
ax.set_ylabel("P(discharged to another facility)"); ax.set_xlabel("Age band")
ax.set_title("A2. Discharge-to-facility rate by race, within age bands (descriptive)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "A2_disp_rate_by_race_age.png"), dpi=150, bbox_inches="tight")
plt.close()

# unadjusetd OR of feature vs Caucasian
d_fit = df[df.race.isin(race_order)].copy()
m_un = smf.logit(f'disp ~ C(race, Treatment(reference="{REF}"))', data=d_fit).fit(disp=0)


def race_ors(res):
    out = {}
    for t in res.params.index:
        if "[T." in t:
            out[t.split("[T.")[1].rstrip("]")] = np.exp(res.params[t])
    return out


or_un = race_ors(m_un)
labels = list(or_un.keys())
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x, [or_un[l] for l in labels], 0.6, color="#c0392b")
ax.axhline(1.0, color="black", ls="--", lw=1)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel(f"OR of discharge-to-facility vs {REF}")
ax.set_title("A3. Odds of discharge to another facility by race (unadjusted)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "A3_disp_or_by_race.png"), dpi=150, bbox_inches="tight")
plt.close()

# Differential meaning: readmit so disp * is_AA
sub = df[df.race.isin([FOCUS, REF])].copy()
sub["is_AA"] = (sub.race == FOCUS).astype(int)
mi = smf.logit("readmit_any ~ disp * is_AA", data=sub).fit(disp=0)
io, iop = np.exp(mi.params["disp:is_AA"]), mi.pvalues["disp:is_AA"]

cells = {}
for race in [FOCUS, REF]:
    for v in [0, 1]:
        s = df[(df.race == race) & (df.disp == v)]
        cells[(race, v)] = prop_ci(s["readmit_any"].sum(), len(s))
fig, ax = plt.subplots(figsize=(8, 5.5))
for race in [FOCUS, REF]:
    ys = [cells[(race, 0)][0], cells[(race, 1)][0]]
    ax.plot([0, 1], ys, "-o", color=palette[race], lw=3, ms=9, label=race)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Home", "Another facility"])
ax.set_ylabel("Readmission rate")
ax.set_title(f"A4. Discharge -> readmission by race (unadjusted)\ninteraction OR {io:.2f} (p={iop:.1e})")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "A4_disp_interaction.png"), dpi=150, bbox_inches="tight")
plt.close()

for r in race_order:
    summary.append(dict(feature="discharged_to_another_facility", race=r,
                        value=A.set_index("race").loc[r, "rate"],
                        or_or_gap_vs_ref_unadjusted=or_un.get(r, np.nan),
                        interaction_note=(f"OR {io:.2f}, p {iop:.2g}" if r == FOCUS else "")))


# mean prior inpatient by race + proportion with >=1
rows = []
for r in race_order:
    g = df[df.race == r]
    m, lo, hi = mean_ci(g["number_inpatient"])
    pany, _, _ = prop_ci((g["number_inpatient"] >= 1).sum(), g["number_inpatient"].notna().sum())
    rows.append(dict(race=r, mean=m, lo=lo, hi=hi, p_any=pany, n=len(g)))
B = pd.DataFrame(rows).sort_values("mean", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(B.race, B["mean"], color=[palette[r] for r in B.race],
            yerr=[B["mean"] - B.lo, B.hi - B["mean"]], capsize=5)
axes[0].axhline(df["number_inpatient"].mean(), color="darkorange", ls="--", lw=2,
                label=f"Overall = {df['number_inpatient'].mean():.2f}")
axes[0].set_ylabel("Mean prior inpatient visits"); axes[0].set_title("Mean prior inpatient visits by race"); axes[0].legend()
axes[1].bar(B.race, B["p_any"], color=[palette[r] for r in B.race])
axes[1].set_ylabel("P(>= 1 prior inpatient visit)"); axes[1].set_title("Share with any prior inpatient visit")
for ax in axes:
    ax.tick_params(axis="x", rotation=15)
fig.suptitle("B1. Prior inpatient utilisation by race", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "B1_ni_mean_by_race.png"), dpi=150, bbox_inches="tight")
plt.close()

# Mean by race x age (descriptive context)
fig, ax = plt.subplots(figsize=(10, 5.5))
for r in race_order:
    ys = []
    for b in age_labels:
        s = df[(df.race == r) & (df.age_band == b)]
        ys.append(s["number_inpatient"].mean() if len(s) >= 50 else np.nan)
    ax.plot(age_labels, ys, "-o", color=palette[r], lw=2, label=r)
ax.set_ylabel("Mean prior inpatient visits"); ax.set_xlabel("Age band")
ax.set_title("B2. Mean prior inpatient visits by race, within age bands (descriptive)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "B2_ni_mean_by_race_age.png"), dpi=150, bbox_inches="tight")
plt.close()

# Readmission rate vs binned prior inpatient count, per race
def ni_bin(x):
    if pd.isna(x):
        return np.nan
    x = int(x)
    return "4+" if x >= 4 else str(x)


df["ni_bin"] = df["number_inpatient"].apply(ni_bin)
bin_order = ["0", "1", "2", "3", "4+"]
fig, ax = plt.subplots(figsize=(10, 6))
for r in race_order:
    ys = []
    for b in bin_order:
        s = df[(df.race == r) & (df.ni_bin == b)]
        ys.append(s["readmit_any"].mean() if len(s) >= 50 else np.nan)
    ax.plot(bin_order, ys, "-o", color=palette[r], lw=2, label=r)
ax.set_xlabel("Prior inpatient visits"); ax.set_ylabel("Readmission rate")
ax.set_title("B3. Readmission vs prior inpatient count, by race\n(diverging slopes = differential meaning)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "B3_ni_readmit_curve_by_race.png"), dpi=150, bbox_inches="tight")
plt.close()

# unadgusted interaction: readmit so number_inpatient * is_AA; ]
mi2 = smf.logit("readmit_any ~ number_inpatient * is_AA", data=sub).fit(disp=0)
io2, io2p = np.exp(mi2.params["number_inpatient:is_AA"]), mi2.pvalues["number_inpatient:is_AA"]

# race gap in the feature itself (extra prior visits vs Caucasian)

gap = smf.ols(f'number_inpatient ~ C(race, Treatment(reference="{REF}"))', data=d_fit).fit()
aa_gap = gap.params.get(f'C(race, Treatment(reference="{REF}"))[T.{FOCUS}]', np.nan)

fig, ax = plt.subplots(figsize=(9, 5.5))
xs = np.array([0, 1, 2, 3, 4])
for race in [FOCUS, REF]:
    ys = []
    for b in bin_order:
        s = df[(df.race == race) & (df.ni_bin == b)]
        ys.append(s["readmit_any"].mean() if len(s) >= 30 else np.nan)
    ax.plot(xs, ys, "-o", color=palette[race], lw=3, ms=9, label=race)
ax.set_xticks(xs); ax.set_xticklabels(bin_order)
ax.set_xlabel("Prior inpatient visits"); ax.set_ylabel("Readmission rate")
ax.set_title(f"B4. Prior-inpatient - readmission slope by race (unadjusted)\n"
             f"per-visit interaction OR {io2:.2f} (p={io2p:.1e}); AA has {aa_gap:+.2f} extra prior visits vs {REF}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "B4_ni_interaction.png"), dpi=150, bbox_inches="tight")
plt.close()

for r in race_order:
    summary.append(dict(feature="number_inpatient", race=r,
                        value=B.set_index("race").loc[r, "mean"],
                        or_or_gap_vs_ref_unadjusted=(aa_gap if r == FOCUS else np.nan),
                        interaction_note=(f"per-visit OR {io2:.2f}, p {io2p:.2g}" if r == FOCUS else "")))

pd.DataFrame(summary).to_csv(os.path.join(OUT, "proxy_audit_unadjusted_summary.csv"), index=False)

# Key point print out

print("=" * 70)
print("PROXY AUDIT (UNADJUSTED) - discharge-to-facility & prior inpatient, by race")
print("=" * 70)
print("\n[A] discharged_to_another_facility (binary)")
print(A[["race", "rate", "n"]].round(3).to_string(index=False))
print(f"  unadjusted OR (AA vs {REF}): {or_un.get(FOCUS, np.nan):.2f}")
print(f"  differential-meaning interaction OR: {io:.2f} (p={iop:.2e})")
print("\n[B] number_inpatient (count)")
print(B[["race", "mean", "p_any", "n"]].round(3).to_string(index=False))
print(f"  AA extra prior visits vs {REF} (unadjusted): {aa_gap:+.3f}")
print(f"  per-visit interaction OR: {io2:.2f} (p={io2p:.2e})")
print("=" * 70)
print("saved figures + proxy_audit_unadjusted_summary.csv to artifacts/proxy_audit_unadjusted/")
