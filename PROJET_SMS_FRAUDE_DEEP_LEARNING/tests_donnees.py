"""
=============================================================================
Script : tests_donnees.py
Mission : Valider la conformité stricte des données générées et de la fonction
          de prétraitement avec des assertions Python.
Auteur : Pair programming assistant & Étudiante
=============================================================================
"""

import os
import sys
import pandas as pd
from preparer_donnees import pretraiter


def tester_pretraiter_unitaire():
    """Vérifie le comportement de la fonction pretraiter sur les cas d'école."""
    print("Test unitaire 1 : Détection combinée MONTANT et URL...")
    t1 = "Vous avez gagné 1 000 000 F, cliquez bit.ly/ab12"
    r1 = pretraiter(t1)
    assert "<MONTANT>" in r1, f"Échec : <MONTANT> attendu dans '{r1}'"
    assert "<URL>" in r1, f"Échec : <URL> attendu dans '{r1}'"
    print("  -> OK :", r1)

    print("Test unitaire 2 : Préservation d'un message légitime usuel...")
    t2 = "Rdv demain à 10h devant la fac"
    r2 = pretraiter(t2)
    assert r2 == t2, f"Échec : Le texte doit rester strictement inchangé. Obtenu : '{r2}'"
    print("  -> OK :", r2)

    print("Test unitaire 3 : Détection d'un numéro de téléphone avec indicatif...")
    t3 = "Appelez le +226 70 12 34 56"
    r3 = pretraiter(t3)
    assert "<TEL>" in r3, f"Échec : <TEL> attendu dans '{r3}'"
    print("  -> OK :", r3)


def tester_fichiers_donnees():
    """Vérifie l'intégrité, la séparation et la stratification des données produites."""
    dossier_data = "data"
    fichiers = {
        "train": os.path.join(dossier_data, "train.csv"),
        "val": os.path.join(dossier_data, "val.csv"),
        "test": os.path.join(dossier_data, "test.csv")
    }

    # Vérification de l'existence des fichiers
    for nom, chemin in fichiers.items():
        assert os.path.exists(chemin), f"Fichier manquant : '{chemin}'"

    # Chargement des trois fichiers CSV
    train_df = pd.read_csv(fichiers["train"], encoding="utf-8-sig")
    val_df = pd.read_csv(fichiers["val"], encoding="utf-8-sig")
    test_df = pd.read_csv(fichiers["test"], encoding="utf-8-sig")

    dfs = {"train": train_df, "val": val_df, "test": test_df}

    # Test 1 : fraude ne contient que 0 et 1 ; aucune valeur manquante dans texte
    print("Test 1 : Vérification des labels 'fraude' et absence de valeurs manquantes...")
    for nom, df in dfs.items():
        valeurs_fraude = set(df["fraude"].unique())
        assert valeurs_fraude.issubset({0, 1}), f"'{nom}' contient des labels invalides : {valeurs_fraude}"
        nb_null_texte = df["texte"].isnull().sum()
        assert nb_null_texte == 0, f"'{nom}' contient {nb_null_texte} valeurs manquantes dans 'texte'"
        nb_vides = (df["texte"].str.strip() == "").sum()
        assert nb_vides == 0, f"'{nom}' contient {nb_vides} textes vides"
    print("  -> OK : Tous les labels sont 0 ou 1, et aucune valeur manquante.")

    # Test 2 : Aucun texte identique entre train, val et test (intersection vide)
    print("Test 2 : Vérification de l'absence de fuite de données (textes identiques)...")
    s_train = set(train_df["texte"])
    s_val = set(val_df["texte"])
    s_test = set(test_df["texte"])

    inter_tr_va = s_train.intersection(s_val)
    inter_tr_te = s_train.intersection(s_test)
    inter_va_te = s_val.intersection(s_test)

    assert len(inter_tr_va) == 0, f"Fuite détectée entre train et val ({len(inter_tr_va)} doublons)"
    assert len(inter_tr_te) == 0, f"Fuite détectée entre train et test ({len(inter_tr_te)} doublons)"
    assert len(inter_va_te) == 0, f"Fuite détectée entre val et test ({len(inter_va_te)} doublons)"
    print("  -> OK : Les trois ensembles ont une intersection strictement vide de textes.")

    # Test 3 : Aucun identifiant id commun aux trois fichiers
    print("Test 3 : Vérification de l'unicité stricte des identifiants 'id'...")
    id_train = set(train_df["id"])
    id_val = set(val_df["id"])
    id_test = set(test_df["id"])

    assert len(id_train.intersection(id_val)) == 0, "IDs communs entre train et val"
    assert len(id_train.intersection(id_test)) == 0, "IDs communs entre train et test"
    assert len(id_val.intersection(id_test)) == 0, "IDs communs entre val et test"
    print("  -> OK : Aucun identifiant commun entre train, val et test.")

    # Test 4 : La proportion de fraudes diffère de moins de 2 points entre les trois fichiers
    print("Test 4 : Vérification de la stratification (< 2 points de différence)...")
    p_train = train_df["fraude"].mean()
    p_val = val_df["fraude"].mean()
    p_test = test_df["fraude"].mean()

    diff_tr_va = abs(p_train - p_val) * 100
    diff_tr_te = abs(p_train - p_test) * 100
    diff_va_te = abs(p_val - p_test) * 100

    print(f"  * Proportion train : {p_train*100:.2f}%")
    print(f"  * Proportion val   : {p_val*100:.2f}% (écart avec train : {diff_tr_va:.3f} pts)")
    print(f"  * Proportion test  : {p_test*100:.2f}% (écart avec train : {diff_tr_te:.3f} pts)")

    assert diff_tr_va < 2.0, f"Écart train/val trop élevé : {diff_tr_va:.3f} points"
    assert diff_tr_te < 2.0, f"Écart train/test trop élevé : {diff_tr_te:.3f} points"
    assert diff_va_te < 2.0, f"Écart val/test trop élevé : {diff_va_te:.3f} points"
    print("  -> OK : La stratification est respectée (écarts < 2 points).")

    # Test 5 : Aucun texte ne contient 'http' ni 'www.' après prétraitement
    print("Test 5 : Vérification de l'absence complète de 'http' et 'www.'...")
    for nom, df in dfs.items():
        has_http = df["texte"].str.contains("http", case=False, regex=False).any()
        has_www = df["texte"].str.contains("www.", case=False, regex=False).any()
        assert not has_http, f"Le fichier '{nom}' contient encore le motif 'http'"
        assert not has_www, f"Le fichier '{nom}' contient encore le motif 'www.'"
    print("  -> OK : Aucun motif 'http' ni 'www.' résiduel dans aucun fichier.")


def main():
    print("=" * 70)
    print("EXÉCUTION DES TESTS DE CONFORMITÉ (tests_donnees.py)")
    print("=" * 70)

    print("\n--- 1. Tests unitaires de la fonction pretraiter ---")
    tester_pretraiter_unitaire()

    print("\n--- 2. Tests de validation des fichiers CSV produits ---")
    tester_fichiers_donnees()

    print("\n" + "=" * 70)
    print("TOUS LES TESTS ONT RÉUSSI AVEC SUCCÈS (0 ERREUR) !")
    print("=" * 70)


if __name__ == "__main__":
    main()
