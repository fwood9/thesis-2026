import os
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

ART = "artifacts"
OUT = os.path.join(ART, "glucose_aa_story")
os.makedirs(OUT, exist_ok=True)

EXP = "glu_measured"
FOCUS = "AfricanAmerican"
REF = "Caucasian"

TEAL_DARK, TEAL_MID, TEAL_LIGHT = "#00695C", "#26A69A", "#80CBC4"
plt.rcParams.update({"axes.edgecolor": "#B0BEC5", "font.size": 12})

df = joblib.load(os.path.join(ART, "df_model.joblib")).copy()
df["readmit_any"] = (df["readmit_class"] >= 1).astype(int)
df["race"] = df["race"].fillna("Missing").astype(str)
df[EXP] = df[EXP].astype(int)

age_bins = [0, 20, 40, 50, 60, 70, 80, 100]
age_labels = ["<20", "20-40", "40-50", "50-60", "60-70", "70-80", "80+"]
df["age_band"] = pd.cut(df["age_numeric"], bins=age_bins, labels=age_labels,
                        right=False, include_lowest=True)


def rate_ci(k, n):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    se = np.sqrt(p * (1 - p) / n)
    return p, max(0, p - 1.96 * se), min(1, p + 1.96 * se)


def cell(dat, race, tested):
    s = dat[(dat.race == race) & (dat[EXP] == tested)]
    return rate_ci(s["readmit_any"].sum(), len(s)), len(s)


def hbar(fname, labels, vals, colors, title, xlabel, xmax):

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    y = range(len(labels))
    ax.barh(y, vals, color=colors, edgecolor="white", height=0.6, zorder=2)
    for i, v in enumerate(vals):
        ax.text(v + xmax * 0.015, i, f"{v:.1%}", va="center", ha="left",
                fontsize=12, fontweight="bold", color="#37474F")
    ax.set_yticks(list(y)); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", ls=":", alpha=0.4, zorder=0)
    ax.tick_params(left=False)
    plt.tight_layout()
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()


stats = {}


p_t, lo_t, hi_t = rate_ci(df[df[EXP] == 1]["readmit_any"].sum(), (df[EXP] == 1).sum())
p_u, lo_u, hi_u = rate_ci(df[df[EXP] == 0]["readmit_any"].sum(), (df[EXP] == 0).sum())

sub = df[df.race.isin([FOCUS, REF])]
rate_focus = df[df.race == FOCUS][EXP].mean()
rate_ref = df[df.race == REF][EXP].mean()
stats["test_rate_AA"] = rate_focus
stats["test_rate_Caucasian"] = rate_ref
stats["test_rate_ratio"] = rate_focus / rate_ref

(rr_aa_t, aa_t_lo, aa_t_hi), n_aa_t = cell(df, FOCUS, 1)
(rr_aa_u, aa_u_lo, aa_u_hi), n_aa_u = cell(df, FOCUS, 0)
(rr_rf_t, rf_t_lo, rf_t_hi), n_rf_t = cell(df, REF, 1)
(rr_rf_u, rf_u_lo, rf_u_hi), n_rf_u = cell(df, REF, 0)
eff_aa = rr_aa_t - rr_aa_u
eff_rf = rr_rf_t - rr_rf_u
stats["effect_AA"] = eff_aa
stats["effect_Caucasian"] = eff_rf

d_int = sub.copy()
d_int["is_AA"] = (d_int.race == FOCUS).astype(int)
m_un = smf.logit(f"readmit_any ~ {EXP} * is_AA", data=d_int).fit(disp=0)
m_adj = smf.logit(f"readmit_any ~ {EXP} * is_AA + age_numeric", data=d_int).fit(disp=0)
inter_term = f"{EXP}:is_AA"
inter_or = np.exp(m_un.params[inter_term])
inter_or_adj = np.exp(m_adj.params[inter_term])
inter_p = m_un.pvalues[inter_term]
inter_p_adj = m_adj.pvalues[inter_term]
stats["interaction_OR"] = inter_or
stats["interaction_OR_ageadj"] = inter_or_adj
stats["interaction_p"] = inter_p
stats["interaction_p_ageadj"] = inter_p_adj

rows = []
for race in [FOCUS, REF]:
    g = df[(df.race == race) & (df.readmit_any == 1)]
    p, lo, hi = rate_ci(g[EXP].sum(), len(g))
    rows.append(dict(race=race, flagged=p, lo=lo, hi=hi, n=len(g)))
flag_cov = pd.DataFrame(rows)
stats["readmit_flag_coverage_AA"] = flag_cov[flag_cov.race == FOCUS]["flagged"].values[0]
stats["readmit_flag_coverage_Caucasian"] = flag_cov[flag_cov.race == REF]["flagged"].values[0]

n_aa = (df.race == FOCUS).sum()
actual_flags = int(df[(df.race == FOCUS)][EXP].sum())
parity_flags = int(round(n_aa * rate_ref))
extra_flags = parity_flags - actual_flags
stats["AA_actual_glucose_flags"] = actual_flags
stats["AA_parity_glucose_flags"] = parity_flags
stats["AA_missing_flags_under_parity"] = extra_flags


# Readmission rate by glucose-testing status (bars)

hbar(os.path.join(OUT, "fig1_readmit_by_testing.png"),
     ["Not tested", "Glucose tested"], [p_u, p_t], [TEAL_MID, TEAL_DARK],
     "Readmission rate by glucose-testing status", "Readmission rate",
     xmax=max(p_u, p_t) * 1.22)


# Glucose-testing rate by race (bars)

hbar(os.path.join(OUT, "fig2_testing_rate_by_race.png"),
     ["African-American", "Caucasian"], [rate_focus, rate_ref], [TEAL_DARK, TEAL_MID],
     "Glucose-testing rate by race", "P(glucose tested)",
     xmax=max(rate_focus, rate_ref) * 1.22)


# Glucose-test effect on readmission by race (slope)

fig, ax = plt.subplots(figsize=(7, 5.2))
ax.plot([0, 1], [rr_aa_u, rr_aa_t], "-o", color=TEAL_DARK, lw=3.2, ms=11,
        label="African-American", zorder=3)
ax.plot([0, 1], [rr_rf_u, rr_rf_t], "-o", color=TEAL_LIGHT, lw=3.2, ms=11,
        label="Caucasian", zorder=3, markeredgecolor=TEAL_MID, markeredgewidth=1.2)
ax.annotate("", xy=(1.07, rr_aa_t), xytext=(1.07, rr_aa_u),
            arrowprops=dict(arrowstyle="<->", color=TEAL_DARK, lw=1.6))
ax.text(1.12, (rr_aa_t + rr_aa_u) / 2, f"+{eff_aa:.1%}", color=TEAL_DARK,
        va="center", fontsize=12, fontweight="bold")
ax.annotate("", xy=(-0.07, rr_rf_t), xytext=(-0.07, rr_rf_u),
            arrowprops=dict(arrowstyle="<->", color=TEAL_MID, lw=1.6))
ax.text(-0.30, (rr_rf_t + rr_rf_u) / 2, f"+{eff_rf:.1%}", color=TEAL_MID,
        va="center", fontsize=12, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Not tested", "Glucose tested"])
ax.set_xlim(-0.5, 1.45); ax.set_ylabel("Readmission rate")
ax.set_title("Glucose-test effect on readmission by race", fontsize=14, fontweight="bold", pad=10)
ax.legend(loc="upper left", frameon=False, fontsize=11.5)
ax.grid(axis="y", ls=":", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_test_race_interaction.png"), dpi=200, bbox_inches="tight")
plt.close()

# Glucose flags: observed vs parity (stacked horizontal bar)

fig, ax = plt.subplots(figsize=(7.5, 2.6))
ax.barh([0], [actual_flags], color=TEAL_LIGHT, edgecolor="white", height=0.5,
        label=f"Observed ({actual_flags:,})", zorder=2)
ax.barh([0], [extra_flags], left=[actual_flags], color=TEAL_DARK, edgecolor="white", height=0.5,
        label=f"Missing ({extra_flags:,})", zorder=2)
ax.text(actual_flags / 2, 0, f"{actual_flags:,}", va="center", ha="center",
        color="#00695C", fontweight="bold", fontsize=12)
ax.text(actual_flags + extra_flags / 2, 0, f"{extra_flags:,}", va="center", ha="center",
        color="white", fontweight="bold", fontsize=12)
ax.set_yticks([0]); ax.set_yticklabels(["African-American\nglucose flags"])
ax.set_xlim(0, parity_flags * 1.02); ax.set_xlabel("Number of glucose flags")
ax.set_title("Glucose flags among African-American patients:\nobserved vs testing parity",
             fontsize=14, fontweight="bold", pad=10)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.grid(axis="x", ls=":", alpha=0.4, zorder=0)
ax.tick_params(left=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.75), ncol=2, frameon=False, fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_flag_parity.png"), dpi=200, bbox_inches="tight")
plt.close()

pd.DataFrame([stats]).T.rename(columns={0: "value"}).to_csv(
    os.path.join(OUT, "glucose_aa_stats.csv"))

print("=" * 60)
print("GLUCOSE / RACE ANALYSIS")
print("=" * 60)
print(f"Tested readmit {p_t:.1%} vs untested {p_u:.1%}")
print(f"Testing rate: AA {rate_focus:.1%} vs Caucasian {rate_ref:.1%} "
      f"(AA = {rate_focus/rate_ref:.0%} of reference)")
print(f"Test effect: AA +{eff_aa:.1%} vs Caucasian +{eff_rf:.1%}; "
      f"interaction OR {inter_or:.2f} (p={inter_p:.2e})")
print(f"Flags among AA readmitted {stats['readmit_flag_coverage_AA']:.1%} "
      f"vs Caucasian {stats['readmit_flag_coverage_Caucasian']:.1%}; "
      f"parity adds ~{extra_flags:,}")
print("=" * 60)
print("saved 4 figures + glucose_aa_stats.csv to", OUT)
