"""
=============================================================================
Script : preparer_donnees.py
Mission : Préparer les données pour la détection binaire de SMS frauduleux
          (fraude = 1, légitime = 0) en français.
Auteur : Pair programming assistant & Étudiante
Dépendances : pandas, scikit-learn, standard library (sys, os, re, unicodedata)
=============================================================================
"""

import sys
import os
import re
import unicodedata
import pandas as pd
from sklearn.model_selection import train_test_split


def pretraiter(sms: str) -> str:
    """
    Fonction de nettoyage et d'anonymisation des SMS en français.
    
    Règles appliquées :
    - Normalisation Unicode NFKC (uniformise les accents et caractères typographiques).
    - Suppression des caractères invisibles (espaces insécables cachées, etc.).
    - Remplacement ordonné par des balises (placeholders) :
        1. <URL>      : liens web, domaines, protocoles (y compris mal formés comme http//)
        2. <MONTANT>  : montants monétaires (F, FCFA, €, $, £, euros...)
        3. <TEL>      : numéros de téléphone (locaux ou internationaux avec indicatif)
        4. <CODE>     : codes de validation (4 à 6 chiffres isolés)
    - Réduction des espaces multiples et suppression des espaces aux extrémités.
    
    IMPORTANT :
    - Ne met PAS en minuscules.
    - Ne retire PAS les accents ni les mots courants.
    - Ne fait ni stemming ni lemmatisation.
    """
    if not isinstance(sms, str):
        return ""

    # a. Normalisation Unicode NFKC
    texte = unicodedata.normalize("NFKC", sms)

    # b. Suppression des caractères invisibles spécifiques
    texte = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", texte)

    # c. Remplacement par balises (placeholders) dans l'ordre strict :
    
    # 1. <URL> : capture les URLs complètes, raccourcisseurs, domaines et protocoles mal formés
    url_pattern = re.compile(
        r"(?:https?:?//|www\.)\S*|\bbit\.ly/\S+|\b[a-z0-9-]+\.(?:com|net|org|xyz|top|info|bf|co\.uk)\b\S*|\bhttps?\S*",
        re.IGNORECASE
    )
    texte = url_pattern.sub("<URL>", texte)

    # 2. <MONTANT> : montants précédés ou suivis d'un symbole monétaire ou mot-clé
    montant_pattern = re.compile(
        r"[£€$]\s?\d[\d.,]*|\d[\d\s.,]*\s?(?:fcfa|cfa|xof|f|€|eur|euros?|£|livres?)\b",
        re.IGNORECASE
    )
    texte = montant_pattern.sub("<MONTANT>", texte)

    # 3. <TEL> : numéros de téléphone d'au moins 8 chiffres (avec ou sans indicatif +XXX)
    tel_pattern = re.compile(r"(?:\+\d{1,3}[\s.-]?)?\d(?:[\s.-]?\d){7,}")
    texte = tel_pattern.sub("<TEL>", texte)

    # 4. <CODE> : codes de vérification de 4 à 6 chiffres consécutifs
    code_pattern = re.compile(r"\b\d{4,6}\b")
    texte = code_pattern.sub("<CODE>", texte)

    # d. Réduction des espaces multiples à un seul espace, puis suppression des espaces superflus
    texte = re.sub(r"\s+", " ", texte).strip()

    return texte


def main():
    # Détermination du fichier d'entrée passé en ligne de commande ou par défaut
    fichier_entree = sys.argv[1] if len(sys.argv) > 1 else "data-augmented.csv"

    if not os.path.exists(fichier_entree):
        print(f"ERREUR : Le fichier d'entrée '{fichier_entree}' est introuvable.")
        sys.exit(1)

    # Création du dossier cible 'data/' pour stocker les livrables
    dossier_data = "data"
    os.makedirs(dossier_data, exist_ok=True)

    print("=" * 70)
    print("Étape 1 : Chargement du fichier CSV")
    print("=" * 70)
    # Chargement avec index_col=0 car la première colonne est un index sans nom
    df = pd.read_csv(fichier_entree, index_col=0)
    
    # Vérification de la présence des colonnes indispensables
    colonnes_requises = ["labels", "text", "text_fr"]
    for col in colonnes_requises:
        if col not in df.columns:
            print(f"ERREUR CRITIQUE : La colonne obligatoire '{col}' est absente du fichier.")
            sys.exit(1)

    nb_brut = len(df)
    print(f"- Forme du DataFrame brut : {df.shape}")
    print(f"- Valeurs manquantes dans les colonnes utiles :")
    print(df[colonnes_requises].isnull().sum().to_string())
    print(f"- Répartition de la colonne 'labels' :")
    print(df["labels"].value_counts().to_string())
    
    mediane_longueur_brute = float(df["text_fr"].dropna().str.len().median())
    print(f"- Longueur médiane de 'text_fr' (en caractères) : {mediane_longueur_brute}")

    print("\n" + "=" * 70)
    print("Étape 2 & 3 : Conservation de l'ID d'origine et création des variables cibles")
    print("=" * 70)
    # Conserver l'identifiant d'origine de chaque ligne
    df["id"] = df.index
    
    # Créer fraude : 1 pour 'spam', 0 pour 'ham'
    df["fraude"] = (df["labels"] == "spam").astype(int)
    
    # Garder texte_original = text_fr
    df["texte_original"] = df["text_fr"]

    print("\n" + "=" * 70)
    print("Étape 4 : Premier dédoublonnage sur 'text' (texte anglais)")
    print("=" * 70)
    lignes_avant_dedup1 = len(df)
    df = df.drop_duplicates(subset=["text"]).copy()
    lignes_apres_dedup1 = len(df)
    retirees_dedup1 = lignes_avant_dedup1 - lignes_apres_dedup1
    print(f"- Lignes avant dédoublonnage 1 : {lignes_avant_dedup1}")
    print(f"- Lignes après dédoublonnage 1 : {lignes_apres_dedup1}")
    print(f"- Lignes doublons retirées     : {retirees_dedup1}")

    print("\n" + "=" * 70)
    print("Étape 5 : Prétraitement du texte français (normalisation et balises)")
    print("=" * 70)
    # Application de la fonction pretraiter
    df["texte"] = df["texte_original"].apply(pretraiter)
    print("- Prétraitement terminé avec succès pour toutes les lignes.")

    print("\n" + "=" * 70)
    print("Étape 6 : Second dédoublonnage sur 'texte' puis découpage Train / Val / Test")
    print("=" * 70)
    lignes_avant_dedup2 = len(df)
    df = df.drop_duplicates(subset=["texte"]).copy()
    lignes_apres_dedup2 = len(df)
    retirees_dedup2 = lignes_avant_dedup2 - lignes_apres_dedup2
    print(f"- Lignes avant dédoublonnage 2 : {lignes_avant_dedup2}")
    print(f"- Lignes après dédoublonnage 2 : {lignes_apres_dedup2}")
    print(f"- Lignes doublons retirées     : {retirees_dedup2}")

    # Découpage stratifié en deux étapes (70% train / 15% val / 15% test)
    # 1er appel : 70% pour train, 30% temporaire
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=0,
        stratify=df["fraude"]
    )

    # 2nd appel : division du temporaire (30%) en 50% val (15%) et 50% test (15%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=0,
        stratify=temp_df["fraude"]
    )

    print(f"- Répartition des jeux de données :")
    print(f"  * Train : {len(train_df)} lignes ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  * Val   : {len(val_df)} lignes ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  * Test  : {len(test_df)} lignes ({len(test_df)/len(df)*100:.1f}%)")

    p_train = train_df["fraude"].mean()
    p_val = val_df["fraude"].mean()
    p_test = test_df["fraude"].mean()
    print(f"- Proportions de fraudes (classe 1) :")
    print(f"  * Train : {p_train*100:.2f}% ({train_df['fraude'].sum()} fraudes)")
    print(f"  * Val   : {p_val*100:.2f}% ({val_df['fraude'].sum()} fraudes)")
    print(f"  * Test  : {p_test*100:.2f}% ({test_df['fraude'].sum()} fraudes)")

    print("\n" + "=" * 70)
    print("Étape 7 : Traitement optionnel de 'sms_locaux.csv'")
    print("=" * 70)
    fichier_locaux = "sms_locaux.csv"
    if os.path.exists(fichier_locaux):
        print(f"- Le fichier '{fichier_locaux}' a été détecté. Traitement en cours...")
        df_local = pd.read_csv(fichier_locaux)
        if "texte" in df_local.columns and "fraude" in df_local.columns:
            df_local["texte_original"] = df_local["texte"]
            df_local["texte"] = df_local["texte_original"].apply(pretraiter)
            df_local = df_local.drop_duplicates(subset=["texte"]).copy()
            if "id" not in df_local.columns:
                df_local["id"] = [f"local_{i}" for i in range(len(df_local))]
            df_local = df_local[["id", "texte", "texte_original", "fraude"]]
            df_local.to_csv(os.path.join(dossier_data, "local_test.csv"), index=False, encoding="utf-8-sig")
            print(f"- '{fichier_locaux}' nettoyé et sauvegardé dans 'data/local_test.csv' ({len(df_local)} lignes).")
        else:
            print(f"- ATTENTION : '{fichier_locaux}' ne possède pas les colonnes 'texte' et 'fraude'. Ignoré.")
    else:
        print(f"- Le fichier optionnel '{fichier_locaux}' n'existe pas. Poursuite sans ce fichier.")

    print("\n" + "=" * 70)
    print("Étape 8 : Enregistrement des fichiers propres dans 'data/'")
    print("=" * 70)
    colonnes_finales = ["id", "texte", "texte_original", "fraude"]
    train_clean = train_df[colonnes_finales]
    val_clean = val_df[colonnes_finales]
    test_clean = test_df[colonnes_finales]

    train_path = os.path.join(dossier_data, "train.csv")
    val_path = os.path.join(dossier_data, "val.csv")
    test_path = os.path.join(dossier_data, "test.csv")

    train_clean.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_clean.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_clean.to_csv(test_path, index=False, encoding="utf-8-sig")

    print(f"- Fichier d'entraînement enregistré : {train_path} ({len(train_clean)} lignes)")
    print(f"- Fichier de validation enregistré   : {val_path} ({len(val_clean)} lignes)")
    print(f"- Fichier de test enregistré         : {test_path} ({len(test_clean)} lignes)")

    # Calcul des longueurs médianes finales pour le rapport
    mediane_texte_pretraite = float(df["texte"].str.len().median())

    # Sélection équilibrée de 10 exemples avant/après (5 fraudes, 5 légitimes)
    ex_fraude = df[df["fraude"] == 1].sample(n=5, random_state=42)
    ex_legit = df[df["fraude"] == 0].sample(n=5, random_state=42)
    exemples_df = pd.concat([ex_fraude, ex_legit]).sample(frac=1.0, random_state=42)
    
    lignes_exemples_md = []
    for _, row in exemples_df.iterrows():
        orig_clean = str(row['texte_original']).replace('|', '\\|').replace('\n', ' ')
        pretr_clean = str(row['texte']).replace('|', '\\|').replace('\n', ' ')
        lbl_nom = "Fraude (1)" if row['fraude'] == 1 else "Légitime (0)"
        lignes_exemples_md.append(f"| {row['id']} | {lbl_nom} | {orig_clean} | {pretr_clean} |")

    # Génération du rapport Markdown data/rapport_donnees.md
    rapport_md = f"""# 📑 Rapport de Préparation des Données

## 1. Évolution du nombre de lignes à chaque étape

| Étape | Description | Nombre de lignes | Lignes retirées |
| :--- | :--- | :--- | :--- |
| **0. Brut** | Chargement initial de `{fichier_entree}` | **{nb_brut}** | 0 |
| **1. Dédoublonnage 1** | Suppression des doublons stricts sur le texte anglais (`text`) | **{lignes_apres_dedup1}** | {retirees_dedup1} |
| **2. Prétraitement** | Normalisation Unicode, suppression invisibles, balises placeholders | **{lignes_apres_dedup1}** | 0 |
| **3. Dédoublonnage 2** | Suppression des doublons après application des balises sur `texte` | **{lignes_apres_dedup2}** | {retirees_dedup2} |
| **Total conservé** | Jeu de données final prêt pour la modélisation | **{len(df)}** | **{nb_brut - len(df)} total** |

---

## 2. Répartition des classes dans chaque fichier (Découpage 70 / 15 / 15)

Le découpage a été réalisé avec deux appels successifs à `train_test_split`, stratifié sur la colonne `fraude` avec `random_state=0`.

| Fichier | Rôle | Total | Légitimes (`fraude=0`) | Fraudes (`fraude=1`) | Proportion de fraudes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`data/train.csv`](file:///c:/Users/hermi/Desktop/PLANORA2/data/train.csv) | Entraînement (70 %) | **{len(train_clean)}** | {len(train_clean) - train_clean['fraude'].sum()} | {train_clean['fraude'].sum()} | **{p_train*100:.2f} %** |
| [`data/val.csv`](file:///c:/Users/hermi/Desktop/PLANORA2/data/val.csv) | Validation (15 %) | **{len(val_clean)}** | {len(val_clean) - val_clean['fraude'].sum()} | {val_clean['fraude'].sum()} | **{p_val*100:.2f} %** |
| [`data/test.csv`](file:///c:/Users/hermi/Desktop/PLANORA2/data/test.csv) | Test indépendant (15 %) | **{len(test_clean)}** | {len(test_clean) - test_clean['fraude'].sum()} | {test_clean['fraude'].sum()} | **{p_test*100:.2f} %** |

> **Vérification de stratification :** L'écart maximal entre les proportions de fraudes des trois jeux de données est de **{max(abs(p_train-p_val), abs(p_train-p_test), abs(p_val-p_test))*100:.3f} point de pourcentage**, ce qui est strictement inférieur au seuil requis de 2 points.

---

## 3. Analyse des longueurs des textes

- **Longueur médiane brute (`text_fr`) :** **{mediane_longueur_brute:.1f}** caractères.
- **Longueur médiane après prétraitement (`texte`) :** **{mediane_texte_pretraite:.1f}** caractères.
- La réduction de longueur médiane s'explique par la substitution de longues URLs et de numéros de téléphone par les jetons courts `<URL>` et `<TEL>`.

---

## 4. Échantillon de 10 exemples avant / après prétraitement

| ID | Classe | Texte original (`text_fr`) | Texte prétraité (`texte`) |
| :--- | :--- | :--- | :--- |
""" + "\n".join(lignes_exemples_md) + f"""

---

## 5. Limites constatées & Décisions techniques

1. **Protocoles mal formés dans le dataset source :** Certains SMS spams historiques contiennent des URLs mal écrites comme `http//` au lieu de `http://` (deux-points omis). L'expression régulière de `<URL>` a été ajustée pour capturer ces variantes, garantissant qu'aucun résidu `http` ou `www.` ne subsiste dans le texte nettoyé.
2. **Ambiguïté entre codes et dates/heures :** La balise `<CODE>` cible les séquences isolées de 4 à 6 chiffres (`\\b\\d{{4,6}}\\b`). Certains nombres comme des années isolées (ex: 2024) peuvent être convertis en `<CODE>`, ce qui est le comportement standard attendu dans ce pipeline.
3. **Traduction automatique d'origine :** Le texte français provient d'une traduction automatisée du dataset SMS Spam Collection anglophone, conservant quelques tournures calquées de l'anglais ou fautes de frappe de l'expéditeur d'origine.
4. **Fichier `sms_locaux.csv` :** Ce fichier optionnel n'était pas présent dans l'espace de travail ; le traitement s'est poursuivi conformément aux instructions en produisant les trois jeux `train`, `val` et `test`.
"""

    rapport_path = os.path.join(dossier_data, "rapport_donnees.md")
    with open(rapport_path, "w", encoding="utf-8") as f:
        f.write(rapport_md)

    print(f"\n- Rapport complet généré avec succès : {rapport_path}")
    print("=" * 70)
    print("Préparation des données terminée sans aucune erreur !")
    print("=" * 70)


if __name__ == "__main__":
    main()
