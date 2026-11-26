import numpy as np
from scipy import stats

def berakna_statistik(df, variabler):
    """Beräknar medelvärde, median, min och max för valda variabler."""
    statistik = df[variabler].agg(["mean", "median", "min", "max"])
    return statistik.round(2)

def berakna_rokarandel(df):
    """Beräknar procentuell andel rökare och icke-rökare."""
    return df["smoker"].value_counts(normalize=True) * 100

def berakna_sjukdomsandel(df):
    """Beräknar andelen personer med sjukdom i datasetet."""
    return df["disease"].mean()

def simulera_sjukdom(andel_sjukdom, antal=1000, seed=123):
    """Simulerar personer och beräknar andelen med sjukdom."""
    np.random.seed(seed)

    simulering = np.random.choice(
        [0, 1],
        size=antal,
        p=[1 - andel_sjukdom, andel_sjukdom]
    )

    return simulering.mean()

def berakna_konfidensintervall(df):
    """Beräknar 95 % konfidensintervall för medelvärdet av systoliskt blodtryck."""

    # Beräkna statistiska mått för systoliskt blodtryck
    medelvarde = df["systolic_bp"].mean()
    standardavvikelse = df["systolic_bp"].std()
    n = len(df["systolic_bp"])

    # Beräkna standardfelet
    standardfel = standardavvikelse / np.sqrt(n)

    # Beräkna 95 % konfidensintervall med normalapproximation
    felmarginal = 1.96 * standardfel

    nedre_grans = medelvarde - felmarginal
    ovre_grans = medelvarde + felmarginal

    return medelvarde, standardavvikelse, standardfel, nedre_grans, ovre_grans


def hypotesprovning_blodtryck(df):
    """Utför ett ensidigt Welch's t-test för systoliskt blodtryck hos rökare och icke-rökare."""

    # Dela upp systoliskt blodtryck efter rökstatus
    bp_rokare = df[df["smoker"] == "Yes"]["systolic_bp"]
    bp_icke_rokare = df[df["smoker"] == "No"]["systolic_bp"]

    # Beräkna medelvärden
    medel_rokare = bp_rokare.mean()
    medel_icke_rokare = bp_icke_rokare.mean()

    # Ensidigt t-test: har rökare högre blodtryck?
    t_stat, p_varde = stats.ttest_ind(
        bp_rokare,
        bp_icke_rokare,
        equal_var=False,
        alternative="greater"
    )

    return medel_rokare, medel_icke_rokare, t_stat, p_varde

def berakna_linjar_regression(df):
    """Beräknar linjär regression mellan ålder och systoliskt blodtryck med NumPy."""

    # Skapa vektorer för ålder (x) och systoliskt blodtryck (y)
    x = df["age"].to_numpy()
    y = df["systolic_bp"].to_numpy()

    # Skapa en matris med en kolumn för intercept och en för ålder
    X = np.column_stack((np.ones(len(x)), x))

    # Beräkna regressionskoefficienterna med minsta kvadratmetoden
    koefficienter, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    intercept = koefficienter[0]
    lutning = koefficienter[1]

    return intercept, lutning

def berakna_r2(df, intercept, lutning):
    """Beräknar förklaringsgraden R² för den linjära regressionsmodellen."""

    # Observerade blodtrycksvärden
    y = df["systolic_bp"].to_numpy()

    # Förutsagda blodtrycksvärden
    y_pred = intercept + lutning * df["age"].to_numpy()

    # Beräkna variationen som modellen inte förklarar
    ss_res = np.sum((y - y_pred) ** 2)

    # Beräkna den totala variationen
    ss_tot = np.sum((y - y.mean()) ** 2)

    # Beräkna förklaringsgraden
    r2 = 1 - ss_res / ss_tot

    return r2