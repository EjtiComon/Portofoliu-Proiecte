import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Importuri Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

##  1. Configurare Globala
# ---------------------------------------------------------------
FRAC = 0.11
SEED = 120
W, H = 128, 128

# --- MODIFICAT ---
TRADUCERI_NUME = {
    "linear": "liniar",
    "nonlinear": "neliniar",

    # Etichete simplificate pentru SVM (pentru nume fisiere)
    "Soft, Wide Influence": "SVM_Soft_Wide",
    "Hard, Local Influence": "SVM_Hard_Local",
    "Compromis (Default)": "SVM_Default"
}

np.random.seed(SEED)
random.seed(SEED)
os.makedirs("data", exist_ok=True)
os.makedirs("out", exist_ok=True)


##  2. Functii Ajutatoare (Generare Imagini & I/O)
# ---------------------------------------------------------------

def rgb_to_int(r, g, b):
    return (r << 16) + (g << 8) + b


def getRGBfromI(val):
    return ((val >> 16) & 255, (val >> 8) & 255, val & 255)


# TASK 1
def genereaza_imagine_originala(cale, latime, inaltime, neliniar=False):
    im = Image.new("RGB", (latime, inaltime), (255, 255, 255))
    pix = im.load()
    cx, cy, r = latime // 2, inaltime // 2, int(0.35 * min(latime, inaltime))
    a = random.random()
    for i in range(latime):
        for j in range(inaltime):
            if neliniar:
                color = (80, 40, 25) if (i - cx) ** 2 + (j - cy) ** 2 <= a * r ** 2 else (63, 50, 70)
            else:
                color = (150, 50, 5) if j >= 1 * i + 0 else (63, 150, 100)
            pix[i, j] = color
    im.save(cale)


def genereaza_imagine_rara(cale_originala, cale_rara, frac):
    gt = Image.open(cale_originala)
    sp = gt.copy()
    pix = sp.load()
    for i in range(sp.width):
        for j in range(sp.height):
            if random.random() > frac:
                pix[i, j] = (255, 255, 255)
    if os.path.exists(cale_rara):
        os.remove(cale_rara)
    sp.save(cale_rara)


# SFARSIT TASK 1

# TASK 2
def citeste_datele(nume_fisier):
    X, y = [], []
    img = Image.open(nume_fisier)
    pix = img.load()
    for i in range(img.width):
        for j in range(img.height):
            if pix[i, j] != (255, 255, 255):
                X.append([i, j])
                y.append(rgb_to_int(*pix[i, j]))
    return np.array(X, float), np.array(y, int)


# SFARSIT TASK 2

# TASK 8
def salveaza_predictia(pred, latime, inaltime, cale):
    im = Image.new("RGB", (latime, inaltime))
    pix = im.load()
    k = 0
    for i in range(latime):
        for j in range(inaltime):
            pix[i, j] = getRGBfromI(pred[k])
            k += 1
    im.save(cale)
    return im


# SFARSIT TASK 8

##  3. Functii Pregatire Date & Modele
# ---------------------------------------------------------------

# TASK 3
def imparte_datele(cale_imagine, test_size=0.3):
    X, y = citeste_datele(cale_imagine)
    return train_test_split(X, y, test_size=test_size, random_state=SEED, stratify=y)


# SFARSIT TASK 3

# TASK 4
def defineste_modelele():
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=SEED),
        "NaiveBayes": Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]),
        "SVM": Pipeline(
            [("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=1, gamma="scale", random_state=SEED))]),
        "ANN": Pipeline([("scaler", StandardScaler()),
                         ("clf", MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=SEED))]),
        "RandomForest": RandomForestClassifier(n_estimators=50, random_state=SEED)
    }
    return models


# SFARSIT TASK 4

##  4. Logica Task-urilor Laboratorului
# ---------------------------------------------------------------

# TASK 5, 6, 7, 9, 10, 13
def ruleaza_evaluarea_de_baza(seturi_date, modele):
    print("\n" + "=" * 50)
    print("=== INCEPERE EVALUARE DE BAZA (TASK 5-10, 13) ===")
    print("=" * 50)

    results = []
    pixeli_totali = np.array([[i, j] for i in range(W) for j in range(H)], float)

    for ds, (Xtr, Xte, ytr, yte) in seturi_date.items():
        ds_ro = TRADUCERI_NUME.get(ds, ds)
        print(f"\n--- Evaluare Set de Date: {ds.upper()} ({ds_ro}) ---")
        X_all, y_all = citeste_datele(f"data/{ds_ro}_rar.png")

        for name, model in modele.items():
            name_ro = TRADUCERI_NUME.get(name, name)

            model.fit(Xtr, ytr)
            ypred = model.predict(Xte)
            acc_test = accuracy_score(yte, ypred)
            y_train_pred = model.predict(Xtr)
            acc_train = accuracy_score(ytr, y_train_pred)

            results.append((ds, name, acc_test))

            print(f"\n[{ds.upper()}] {name}: acuratete test={acc_test:.3f}, acuratete train={acc_train:.3f}")
            print(classification_report(yte, ypred, zero_division=0))

            if name in ["DecisionTree", "SVM", "RandomForest"]:
                print(f"--- [Studiu CV pentru {name} pe {ds.upper()}] ---")
                k_values = [3, 5, 10]
                for k in k_values:
                    scores = cross_val_score(model, X_all, y_all, cv=k, scoring='accuracy')
                    mean_acc = scores.mean()
                    print(f"  k={k}: Media={mean_acc:.3f} | Comparatie: CV ({mean_acc:.3f}) vs. Split ({acc_test:.3f})")

            pred_full = model.predict(pixeli_totali)
            nume_fisier_salvat = f"out/{ds_ro}_{name_ro}.png"
            salveaza_predictia(pred_full, W, H, nume_fisier_salvat)
            print(f"Imaginea restaurată salvată: {nume_fisier_salvat}")

    return pd.DataFrame(results, columns=["Set", "Model", "Acuratete"])


# SFARSIT TASK 5, 6, 7, 9, 10, 13

# TASK 11
def ruleaza_studiu_manual_svm(seturi_date):
    print("\n" + "=" * 50)
    print("=== 🔬 STUDIU MANUAL HIPERPARAMETRI SVM (TASK 11) ===")
    print("=" * 50)

    param_combinations = [
        {"C": 0.1, "gamma": 0.01, "label": "Soft, Wide Influence"},
        {"C": 10, "gamma": 1, "label": "Hard, Local Influence"},
        {"C": 1, "gamma": 'scale', "label": "Compromis (Default)"}
    ]
    manual_results = []
    pixeli_totali = np.array([[i, j] for i in range(W) for j in range(H)], float)

    for ds, (Xtr, Xte, ytr, yte) in seturi_date.items():
        ds_ro = TRADUCERI_NUME.get(ds, ds)
        print(f"\n--- SETUL: {ds.upper()} ({ds_ro}) ---")

        for params in param_combinations:
            svm_model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=params["C"], gamma=params["gamma"], random_state=SEED))
            ])
            svm_model.fit(Xtr, ytr)
            ypred = svm_model.predict(Xte)
            acc = accuracy_score(yte, ypred)
            manual_results.append((ds, params["label"], acc))
            print(f"  > C={params['C']}, Gamma={params['gamma']}: Acuratete={acc:.3f} ({params['label']})")

            descriere_ro = TRADUCERI_NUME.get(params["label"], params["label"])
            nume_fisier_salvat = f"out/{ds_ro}_{descriere_ro}.png"

            pred_full = svm_model.predict(pixeli_totali)
            salveaza_predictia(pred_full, W, H, nume_fisier_salvat)
            print(f"  Imagine salvată: {nume_fisier_salvat}")

    df_manual = pd.DataFrame(manual_results, columns=["Set", "Configuratie", "Acuratete"])
    print("\n=== Rezumat Studiu Manual SVM ===")
    print(df_manual.pivot(index="Configuratie", columns="Set", values="Acuratete"))


# SFARSIT TASK 11

# TASK 12
def ruleaza_optimizare_svm_grid_search(set_date_liniar):
    print("\n" + "=" * 50)
    print("=== GRID SEARCH CV OPTIMIZARE SVM (TASK 12) ===")
    print("=" * 50)

    Xtr_linear, _, ytr_linear, _ = set_date_liniar
    param_grid = {
        'clf__C': [0.1, 1, 10],
        'clf__gamma': [0.01, 0.1, 'scale', 1]
    }
    pipeline_gs = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel='rbf', random_state=SEED))
    ])

    print("Incepe Grid Search pe setul LINIAR (dureaza cateva secunde)...")
    grid_search = GridSearchCV(pipeline_gs, param_grid, cv=5, scoring='accuracy', verbose=0)
    grid_search.fit(Xtr_linear, ytr_linear)

    print("\n--- Rezultate Grid Search ---")
    print(f"Parametri optimi: {grid_search.best_params_}")
    print(f"Cea mai buna acuratete CV: {grid_search.best_score_:.4f}")

    optimal_model = grid_search.best_estimator_
    pixeli_totali = np.array([[i, j] for i in range(W) for j in range(H)], float)
    pred_optimal = optimal_model.predict(pixeli_totali)

    nume_fisier_salvat = "out/liniar_SVM_Optim_GridSearch.png"
    salveaza_predictia(pred_optimal, W, H, nume_fisier_salvat)
    print(f"Imaginea optima salvata: {nume_fisier_salvat}")


# SFARSIT TASK 12

# TASK 15
def ruleaza_studiu_densitate_date():
    print("\n" + "=" * 50)
    print("=== STUDIU DENSITATE DATE (TASK 15) ===")
    print("=" * 50)

    FRAC_INITIAL = 0.1
    FRAC_DENS = 0.3  # Am verificat, aceasta este varianta corecta (cu S)
    model_name = "RandomForest"
    model_name_ro = TRADUCERI_NUME.get(model_name, model_name)
    clf = RandomForestClassifier(n_estimators=50, random_state=SEED)
    ds_list = ["linear", "nonlinear"]
    data_results = []

    for ds in ds_list:
        ds_ro = TRADUCERI_NUME.get(ds, ds)
        print(f"\n--- Setul {ds.upper()} ({ds_ro}) ---")

        cale_rara_densa = f"data/{ds_ro}_rar_dens.png"
        cale_rara_initiala = f"data/{ds_ro}_rar_initial.png"
        cale_originala = f"data/{ds_ro}_original.png"

        genereaza_imagine_rara(cale_originala, cale_rara_densa, FRAC_DENS)
        genereaza_imagine_rara(cale_originala, cale_rara_initiala, FRAC_INITIAL)

        Xtr_i, Xte_i, ytr_i, yte_i = imparte_datele(cale_rara_initiala)
        clf.fit(Xtr_i, ytr_i)
        acc_initial = accuracy_score(yte_i, clf.predict(Xte_i))

        Xtr_d, Xte_d, ytr_d, yte_d = imparte_datele(cale_rara_densa)
        clf.fit(Xtr_d, ytr_d)
        acc_dense = accuracy_score(yte_d, clf.predict(Xte_d))

        data_results.append((ds, FRAC_INITIAL, acc_initial))
        data_results.append((ds, FRAC_DENS, acc_dense))

        print(f"  > FRAC={FRAC_INITIAL:.1f} (10% date): Ac. Test={acc_initial:.4f}")
        print(f"  > FRAC={FRAC_DENS:.1f} (30% date): Ac. Test={acc_dense:.4f}")

        pixeli_totali = np.array([[i, j] for i in range(W) for j in range(H)], float)
        pred_full = clf.predict(pixeli_totali)
        nume_fisier_salvat = f"out/{ds_ro}_{model_name_ro}_Densitate_{int(FRAC_DENS * 100)}pct.png"
        salveaza_predictia(pred_full, W, H, nume_fisier_salvat)
        print(f"  Imagine salvată (densitate 30%): {nume_fisier_salvat}")

    df_data = pd.DataFrame(data_results, columns=["Set", "Frac", "Acuratete Test"])
    print("\n=== Rezumat Studiu Densitate Date ===")
    print(df_data.pivot(index="Frac", columns="Set", values="Acuratete Test"))


# SFARSIT TASK 15

# TASK 16
def ruleaza_studiu_raport_impartire():
    print("\n" + "=" * 50)
    print("=== STUDIU RAPORT ÎMPĂRȚIRE TRAIN/TEST (TASK 16) ===")
    print("=" * 50)

    model_name = "RandomForest"
    model_name_ro = TRADUCERI_NUME.get(model_name, model_name)
    clf = RandomForestClassifier(n_estimators=50, random_state=SEED)
    split_ratios_test = [0.3, 0.2, 0.5]
    split_results = []
    ds_study = "linear"
    ds_study_ro = TRADUCERI_NUME.get(ds_study, ds_study)

    print(f"Model: {model_name} pe setul {ds_study.upper()}")
    pixeli_totali = np.array([[i, j] for i in range(W) for j in range(H)], float)

    for ts in split_ratios_test:
        train_pct = 1.0 - ts
        Xtr, Xte, ytr, yte = imparte_datele(f"data/{ds_study_ro}_rar.png", test_size=ts)

        clf.fit(Xtr, ytr)
        ypred = clf.predict(Xte)
        acc_test = accuracy_score(yte, ypred)
        acc_train = accuracy_score(ytr, clf.predict(Xtr))

        split_label = f"{int(train_pct * 100)}% Train / {int(ts * 100)}% Test"
        split_results.append((split_label, acc_train, acc_test))
        print(f"  > Împărțire {split_label}: Ac. Train={acc_train:.4f}, Ac. Test={acc_test:.4f}")

        pred_full = clf.predict(pixeli_totali)
        nume_fisier_salvat = f"out/{ds_study_ro}_{model_name_ro}_Impartire_{int(train_pct * 100)}Train.png"
        salveaza_predictia(pred_full, W, H, nume_fisier_salvat)
        print(f"  Imagine salvată: {nume_fisier_salvat}")

    df_split = pd.DataFrame(split_results, columns=["Împărțire", "Acuratețe Train", "Acuratețe Test"])
    print("\n=== Rezumat Studiu Împărțire ===")
    print(df_split.set_index("Împărțire"))


# SFARSIT TASK 16


##  5. Functii de Vizualizare
# ---------------------------------------------------------------

def afiseaza_imagini_initiale():
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    titles = ["Liniara originală", "Liniara rară (date intrare)", "Neliniara originală",
              "Neliniara rară (date intrare)"]
    images = [
        Image.open("data/liniar_original.png"),
        Image.open("data/liniar_rar.png"),
        Image.open("data/neliniar_original.png"),
        Image.open("data/neliniar_rar.png")
    ]
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(titles[i])
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# TASK 14
def afiseaza_grafice_comparative(results_df, modele_dict):
    for ds in ["linear", "nonlinear"]:
        ds_ro = TRADUCERI_NUME.get(ds, ds)
        fig, axes = plt.subplots(1, len(modele_dict), figsize=(14, 3))
        fig.suptitle(f"Imagini restaurate - set {ds_ro.capitalize()}", fontsize=14)

        for i, name in enumerate(modele_dict.keys()):
            name_ro = TRADUCERI_NUME.get(name, name)
            cale_fisier = f"out/{ds_ro}_{name_ro}.png"
            try:
                im = Image.open(cale_fisier)
                axes[i].imshow(im)
                axes[i].set_title(name_ro)
                axes[i].axis("off")
            except FileNotFoundError:
                print(f"Atenție: Nu am găsit fișierul {cale_fisier}")
                axes[i].axis("off")
        plt.tight_layout()
        plt.show()

    print("\n" + "=" * 50)
    print("=== REZUMAT COMPARATIV FINAL (TASK 14) ===")
    print("=" * 50)

    print("\n=== Rezumat Acuratețe (Test) ===")
    pivot_df = results_df.pivot(index="Model", columns="Set", values="Acuratete")
    print(pivot_df)

    pivot_df.plot(kind="bar", figsize=(10, 6))
    plt.title("Acuratețe clasificatori pe imagini liniar vs neliniar separabile")
    plt.ylabel("Acuratețe Test")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# SFARSIT TASK 14


##  6. Executie Principala
# ---------------------------------------------------------------

def main():
    print("TASK 1: Se genereaza imaginile...")
    genereaza_imagine_originala("data/liniar_original.png", W, H, neliniar=False)
    genereaza_imagine_originala("data/neliniar_original.png", W, H, neliniar=True)
    genereaza_imagine_rara("data/liniar_original.png", "data/liniar_rar.png", FRAC)
    genereaza_imagine_rara("data/neliniar_original.png", "data/neliniar_rar.png", FRAC)

    seturi_de_date = {
        "linear": imparte_datele("data/liniar_rar.png"),
        "nonlinear": imparte_datele("data/neliniar_rar.png")
    }

    modele = defineste_modelele()

    afiseaza_imagini_initiale()

    rezultate_df = ruleaza_evaluarea_de_baza(seturi_de_date, modele)

    ruleaza_studiu_manual_svm(seturi_de_date)

    ruleaza_optimizare_svm_grid_search(seturi_de_date["linear"])

    ruleaza_studiu_densitate_date()

    ruleaza_studiu_raport_impartire()

    afiseaza_grafice_comparative(rezultate_df, modele)


if __name__ == "__main__":
    main()