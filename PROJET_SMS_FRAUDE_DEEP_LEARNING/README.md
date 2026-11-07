# 🛡️ Projet 2 : Détection de SMS Frauduleux par Deep Learning

Ce dossier regroupe l'intégralité du projet de détection de SMS frauduleux (fraude = 1, légitime = 0) en français : données nettoyées, scripts de prétraitement, suite de tests unitaires, modèle neuronale BiLSTM entraîné, scripts d'inférence et rapports techniques.

---

## 📁 Structure du Dossier

```text
PROJET_SMS_FRAUDE_DEEP_LEARNING/
│
├── README.md                      # Guide d'utilisation complet (ce fichier)
├── requirements.txt               # Dépendances Python du projet
├── lancer_projet.bat              # Lance la préparation, les tests et l'inférence en 1 clic
│
├── 📜 Scripts Python principaux :
│   ├── preparer_donnees.py        # Étape 1 : Nettoyage, balisage NFKC/URL/MONTANT/TEL/CODE, split 70/15/15
│   ├── tests_donnees.py           # Étape 2 : Vérification automatique des données (assertions)
│   ├── modele_deep_learning.py    # Étape 3 : Entraînement et évaluation du BiLSTM PyTorch
│   └── tester_mon_sms.py          # Étape 4 : Démonstration interactive pour tester vos propres SMS
│
├── 🧠 Modèle & Vocabulaire :
│   ├── modele_sms_bilstm.pt       # Poids du modèle Deep Learning (Accuracy: 98.96%, F1: 95.60%)
│   └── vocabulaire_sms.json       # Dictionnaire des 5 000 tokens textuels
│
├── 📊 Données officielles du projet (data/) :
│   ├── train.csv                  # 70 % d'entraînement (3 584 lignes)
│   ├── val.csv                    # 15 % de validation (768 lignes)
│   ├── test.csv                   # 15 % de test indépendant (769 lignes)
│   ├── rapport_donnees.md         # Rapport d'analyse et de préparation des données
│   └── rapport_deep_learning.md   # Rapport technique complet de modélisation
│
├── 📥 Données brutes :
│   └── data-augmented.csv         # Fichier d'entrée initial (5 572 lignes)
│
└── 🎁 Bonus - Dataset bancaire complémentaire (dataset_complementaire_banque/) :
    └── Données traduites spécialisées en arnaques bancaires / Mobile Money (M-Pesa, BIM, BCI)
```

---

## 🚀 Guide d'Exécution Rapide

### 0. Installer les dépendances
```bash
python -m pip install -r requirements.txt
```
> Tkinter est fourni avec la plupart des installations Python sous Windows ; il n'est pas installé via `pip`.

### 1. Préparer les données
```bash
python preparer_donnees.py data-augmented.csv
```
> Nettoie les textes, remplace les entités sensibles par `<URL>`, `<MONTANT>`, `<TEL>`, `<CODE>`, effectue le double dédoublonnage et crée les jeux `train.csv`, `val.csv` et `test.csv` dans `data/`.

### 2. Lancer les tests de conformité
```bash
python tests_donnees.py
```
> Vérifie l'absence de valeurs manquantes, la stricte absence de fuite de données entre train/val/test, la stratification à < 2 points d'écart, et les tests unitaires de la fonction de prétraitement.

### 3. Entraîner et évaluer le modèle Deep Learning
```bash
python modele_deep_learning.py
```
> Entraîne le réseau BiLSTM sur CPU, sauvegarde le meilleur modèle dans `modele_sms_bilstm.pt` et affiche le rapport de classification complet sur le jeu de test.

### 4. Tester n'importe quel SMS en direct (Inférence interactive)
```bash
python tester_mon_sms.py "Félicitations ! Vous avez gagné 500 000 F, cliquez sur bit.ly/promo"
```
> Affiche instantanément la version prétraitée, le niveau de risque estimé (de 0 % à 100 %) et le verdict (`[ALERTE FRAUDE]` ou `[MESSAGE LEGITIME]`).

### 5. Utiliser l'interface graphique
```bash
python interface_graphique.py
```
> Ouvre une fenêtre pour saisir ou coller un SMS et afficher le verdict, le risque estimé et le texte prétraité. Sous Windows, vous pouvez aussi double-cliquer sur `lancer_interface.bat`.

L'interface réutilise le modèle et le vocabulaire déjà présents dans le dossier du projet. Elle nécessite Python avec Tkinter et les dépendances du modèle installées. La prédiction est une aide et ne garantit pas qu'un SMS est sûr.

---

## 📈 Résultats du Modèle sur le Jeu de Test (`data/test.csv`)

| Métrique | Score | Interprétation |
| :--- | :--- | :--- |
| **Exactitude (Accuracy)** | **98,96 %** | 761 SMS bien classés sur 769 |
| **Score F1 (Fraude)** | **95,60 %** | Équilibre optimal entre précision et détection |
| **Précision (Fraude)** | **98,86 %** | 1 seul faux positif sur 675 messages légitimes |
| **Rappel (Fraude)** | **92,55 %** | 87 fraudes neutralisées sur 94 |
| **ROC-AUC** | **96,20 %** | Séparabilité presque parfaite des deux classes |
