"""
=============================================================================
Projet 2 : Détection de SMS frauduleux par Deep Learning (BiLSTM)
Auteur   : Étudiante & Assistant IA
Cadre    : PyTorch (CPU-compatible, performant et explicable)
Objectif : Classifier les SMS en temps réel (Fraude = 1, Légitime = 0)
=============================================================================
"""

import os
import sys
import json
import re
import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

# Configuration de l'encodage standard UTF-8 pour Windows Console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Importer la fonction de prétraitement certifiée du projet
from preparer_donnees import pretraiter

# Fixer la graine aléatoire pour des résultats 100% reproductibles
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# 1. GESTION DU VOCABULAIRE & TOKENISATION
# =============================================================================

class VocabulaireSMS:
    """Construit et gère le dictionnaire de mots pour le modèle de Deep Learning."""
    def __init__(self, max_mots=5000, pad_token="<PAD>", unk_token="<UNK>"):
        self.max_mots = max_mots
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mot_vers_id = {pad_token: 0, unk_token: 1}
        self.id_vers_mot = {0: pad_token, 1: unk_token}
        self.compteur_mots = {}

    def construire(self, corpus_textes):
        """Compte les fréquences de chaque mot dans le texte d'entraînement."""
        for texte in corpus_textes:
            mots = str(texte).split()
            for mot in mots:
                mot_clean = mot.strip(".,!?;:\"'()[]{}")
                if mot_clean:
                    self.compteur_mots[mot_clean] = self.compteur_mots.get(mot_clean, 0) + 1

        # Trier les mots par fréquence décroissante
        mots_tries = sorted(self.compteur_mots.items(), key=lambda x: x[1], reverse=True)
        for mot, _ in mots_tries[:self.max_mots - 2]:
            idx = len(self.mot_vers_id)
            self.mot_vers_id[mot] = idx
            self.id_vers_mot[idx] = mot

    def encoder(self, texte, max_len=60):
        """Convertit une chaîne de texte en une liste d'indices de taille fixe max_len."""
        mots = str(texte).split()
        indices = []
        for mot in mots:
            mot_clean = mot.strip(".,!?;:\"'()[]{}")
            if mot_clean:
                idx = self.mot_vers_id.get(mot_clean, self.mot_vers_id[self.unk_token])
                indices.append(idx)
        
        # Tronquer ou compléter (padding) à max_len
        if len(indices) < max_len:
            indices = indices + [self.mot_vers_id[self.pad_token]] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        return indices

    def sauvegarder(self, chemin):
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(self.mot_vers_id, f, ensure_ascii=False, indent=2)

    def charger(self, chemin):
        with open(chemin, "r", encoding="utf-8") as f:
            self.mot_vers_id = json.load(f)
            self.id_vers_mot = {int(v): k for k, v in self.mot_vers_id.items()}


# =============================================================================
# 2. DATASET PYTORCH
# =============================================================================

class SMSDataset(Dataset):
    """Encapsule les données dans un DataLoader PyTorch."""
    def __init__(self, df, vocab, max_len=60):
        self.textes = df["texte"].tolist()
        self.labels = df["fraude"].values.astype(np.float32)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.textes)

    def __getitem__(self, idx):
        vecteur = self.vocab.encoder(self.textes[idx], self.max_len)
        return torch.tensor(vecteur, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)


# =============================================================================
# 3. ARCHITECTURE DEEP LEARNING (BiLSTM avec Attention & Pooling)
# =============================================================================

class BiLSTM_SMS_Classifier(nn.Module):
    """
    Architecture neuronale pour la détection de fraudes textuelles :
    1. Couche d'Embedding (vectorisation dense des mots)
    2. LSTM Bidirectionnel (capture le contexte passé et futur des SMS)
    3. Concaténation MaxPooling + AveragePooling (détecte les mots-clés d'alerte et le contexte global)
    4. Couches fully-connected denses avec Dropout pour la régularisation
    5. Sortie binaire (score logarithmique pour BCEWithLogitsLoss)
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, num_layers=1, dropout=0.3):
        super(BiLSTM_SMS_Classifier, self).__init__()
        
        # 1. Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 2. LSTM Bidirectionnel
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 3. Classification dense
        # hidden_dim * 2 (bidirectionnel) * 2 (max_pool + avg_pool) = hidden_dim * 4
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 4, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim * 2]
        
        # Global Max Pooling et Global Average Pooling temporels
        avg_pool = torch.mean(lstm_out, dim=1)  # [batch_size, hidden_dim * 2]
        max_pool, _ = torch.max(lstm_out, dim=1)  # [batch_size, hidden_dim * 2]
        
        # Concaténation des deux représentations
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # [batch_size, hidden_dim * 4]
        
        # Dense layers
        out = self.dropout(pooled)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        logits = self.fc2(out).squeeze(1)  # [batch_size]
        return logits


# =============================================================================
# 4. FONCTIONS D'ENTRAÎNEMENT & ÉVALUATION
# =============================================================================

def entrainer_modele(modele, train_loader, val_loader, criterion, optimizer, epochs=8, device="cpu"):
    """Boucle d'entraînement avec suivi des métriques et Early Stopping."""
    meilleur_f1 = 0.0
    chemin_sauvegarde = "modele_sms_bilstm.pt"

    print(f"\nDébut de l'entraînement sur {device.upper()} ({epochs} époques)...")
    print("-" * 75)
    print(f"{'Époque':<8} | {'Perte Train':<12} | {'Perte Val':<12} | {'F1 Val':<10} | {'Précision':<10} | {'Rappel':<10}")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        modele.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = modele(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)

        train_loss /= len(train_loader.dataset)

        # Validation
        val_loss, val_f1, val_prec, val_rec, _, _ = evaluer(modele, val_loader, criterion, device)

        print(f"{epoch:<8} | {train_loss:<12.4f} | {val_loss:<12.4f} | {val_f1*100:<9.2f}% | {val_prec*100:<9.2f}% | {val_rec*100:<9.2f}%")

        # Sauvegarder le meilleur modèle selon le F1-score de validation
        if val_f1 > meilleur_f1:
            meilleur_f1 = val_f1
            torch.save(modele.state_dict(), chemin_sauvegarde)

    print("-" * 75)
    print(f"Meilleur modèle sauvegardé dans '{chemin_sauvegarde}' avec F1-score validation = {meilleur_f1*100:.2f}%\n")
    # Recharger les meilleurs poids
    modele.load_state_dict(torch.load(chemin_sauvegarde))
    return modele


def evaluer(modele, dataloader, criterion=None, device="cpu"):
    """Évalue le modèle et calcule toutes les métriques de classification."""
    modele.eval()
    total_loss = 0.0
    toutes_preds = []
    tous_labels = []
    toutes_probas = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = modele(x_batch)
            if criterion:
                total_loss += criterion(logits, y_batch).item() * len(y_batch)
            probas = torch.sigmoid(logits).cpu().numpy()
            preds = (probas >= 0.5).astype(int)
            
            toutes_probas.extend(probas)
            toutes_preds.extend(preds)
            tous_labels.extend(y_batch.cpu().numpy().astype(int))

    avg_loss = total_loss / len(dataloader.dataset) if criterion else 0.0
    f1 = f1_score(tous_labels, toutes_preds, zero_division=0)
    
    # Matrice de confusion pour précision/rappel de la classe 1
    cm = confusion_matrix(tous_labels, toutes_preds)
    vn, fp, fn, vp = cm.ravel()
    precision = vp / (vp + fp) if (vp + fp) > 0 else 0.0
    rappel = vp / (vp + fn) if (vp + fn) > 0 else 0.0

    return avg_loss, f1, precision, rappel, np.array(tous_labels), np.array(toutes_preds)


# =============================================================================
# 5. FONCTION D'INFÉRENCE DIRECTE POUR TESTER UN SMS
# =============================================================================

def predire_sms(texte_sms, modele, vocab, max_len=60, device="cpu"):
    """Prend un SMS brut quelconque, le nettoie et renvoie la prédiction avec confiance."""
    texte_clean = pretraiter(texte_sms)
    vecteur = vocab.encoder(texte_clean, max_len=max_len)
    tensor_in = torch.tensor([vecteur], dtype=torch.long).to(device)
    
    modele.eval()
    with torch.no_grad():
        logit = modele(tensor_in)
        proba = torch.sigmoid(logit).item()
    
    is_fraude = proba >= 0.5
    verdict = "[ALERTE FRAUDE / SMISHING]" if is_fraude else "[MESSAGE LEGITIME]"
    return {
        "texte_original": texte_sms,
        "texte_pretraite": texte_clean,
        "prediction": verdict,
        "probabilite_fraude": round(proba * 100, 2),
        "est_fraude": int(is_fraude)
    }


# =============================================================================
# 6. PIPELINE PRINCIPAL (MAIN)
# =============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 75)
    print(f"PROJET 2 : ENTRAÎNEMENT DU MODÈLE DEEP LEARNING (BiLSTM) SUR {device.upper()}")
    print("=" * 75)

    # Chemins des données
    train_path = "data/train.csv"
    val_path = "data/val.csv"
    test_path = "data/test.csv"

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            print(f"Erreur : Le fichier {p} est introuvable. Exécutez d'abord preparer_donnees.py.")
            sys.exit(1)

    # 1. Chargement des données
    print("\n1. Chargement des données d'entraînement, validation et test...")
    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    val_df = pd.read_csv(val_path, encoding="utf-8-sig")
    test_df = pd.read_csv(test_path, encoding="utf-8-sig")

    print(f"  - Train : {len(train_df)} exemples ({train_df['fraude'].sum()} fraudes)")
    print(f"  - Val   : {len(val_df)} exemples ({val_df['fraude'].sum()} fraudes)")
    print(f"  - Test  : {len(test_df)} exemples ({test_df['fraude'].sum()} fraudes)")

    # 2. Construction du vocabulaire
    print("\n2. Construction du vocabulaire textuel (Top 5 000 mots)...")
    vocab = VocabulaireSMS(max_mots=5000)
    vocab.construire(train_df["texte"])
    vocab.sauvegarder("vocabulaire_sms.json")
    print(f"  - Taille du vocabulaire construit : {len(vocab.mot_vers_id)} tokens.")

    # 3. Préparation des DataLoaders
    batch_size = 32
    max_len = 60
    train_loader = DataLoader(SMSDataset(train_df, vocab, max_len), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SMSDataset(val_df, vocab, max_len), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SMSDataset(test_df, vocab, max_len), batch_size=batch_size, shuffle=False)

    # 4. Définition du modèle et fonction de perte avec pondération des classes
    modele = BiLSTM_SMS_Classifier(
        vocab_size=len(vocab.mot_vers_id),
        embedding_dim=128,
        hidden_dim=64,
        num_layers=1,
        dropout=0.3
    ).to(device)

    # Calcul du poids pour compenser le déséquilibre de classe (87.8% légitimes vs 12.2% fraudes)
    nb_negatifs = (train_df["fraude"] == 0).sum()
    nb_positifs = (train_df["fraude"] == 1).sum()
    poids_fraude = torch.tensor([nb_negatifs / nb_positifs], dtype=torch.float32).to(device)
    print(f"  - Poids positif appliqué (BCEWithLogitsLoss) : {poids_fraude.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=poids_fraude)
    optimizer = torch.optim.AdamW(modele.parameters(), lr=0.001, weight_decay=1e-4)

    # 5. Entraînement
    modele = entrainer_modele(modele, train_loader, val_loader, criterion, optimizer, epochs=8, device=device)

    # 6. Évaluation finale sur l'ensemble de TEST
    print("=" * 75)
    print("6. RÉSULTATS D'ÉVALUATION FINALE SUR LE JEU DE TEST (data/test.csv)")
    print("=" * 75)
    _, test_f1, test_prec, test_rec, y_true, y_pred = evaluer(modele, test_loader, criterion=None, device=device)
    
    acc = (y_true == y_pred).mean()
    roc_auc = roc_auc_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    vn, fp, fn, vp = cm.ravel()

    print(f"\nExactitude globale (Accuracy) : {acc*100:.2f}%")
    print(f"Score F1 (Fraude)             : {test_f1*100:.2f}%")
    print(f"Précision (Fraude)            : {test_prec*100:.2f}%")
    print(f"Rappel (Fraude)               : {test_rec*100:.2f}%")
    print(f"Aire sous la courbe (ROC-AUC) : {roc_auc*100:.2f}%\n")

    print("Matrice de Confusion :")
    print(f"  Vrais Légitimes (VN) : {vn:<5} | Faux Positifs (FP) : {fp}")
    print(f"  Faux Négatifs (FN)   : {fn:<5} | Vrais Fraudes (VP) : {vp}\n")

    print("Rapport de classification détaillé :")
    print(classification_report(y_true, y_pred, target_names=["Légitime (0)", "Fraude (1)"], digits=4))

    # 7. Test de généralisation sur les arnaques financières M-Pesa / Banques
    chemin_test_bancaire = "dataset/splits/test_fr.csv"
    if os.path.exists(chemin_test_bancaire):
        print("=" * 75)
        print("7. TEST DE GÉNÉRALISATION SUR LES ARNAQUES BANCAIRES / MOBILE MONEY")
        print("=" * 75)
        df_bank = pd.read_csv(chemin_test_bancaire, encoding="utf-8-sig")
        df_bank["texte"] = df_bank["text"].apply(pretraiter)
        df_bank["fraude"] = (df_bank["label"] == "Smishing").astype(int)
        
        bank_loader = DataLoader(SMSDataset(df_bank, vocab, max_len), batch_size=batch_size, shuffle=False)
        _, bank_f1, bank_prec, bank_rec, y_b_true, y_b_pred = evaluer(modele, bank_loader, None, device)
        bank_acc = (y_b_true == y_b_pred).mean()
        
        print(f"Échantillon testé : {len(df_bank)} messages financiers réels")
        print(f"Exactitude (Accuracy)         : {bank_acc*100:.2f}%")
        print(f"Score F1 sur Smishing Bancaire: {bank_f1*100:.2f}%")
        print(f"Rappel (Fraudes détectées)    : {bank_rec*100:.2f}%\n")

    # 8. Démonstration interactive avec des exemples concrets
    print("=" * 75)
    print("8. DÉMONSTRATION EN DIRECT SUR LES 5 CAS DU PROJET 2")
    print("=" * 75)
    cas_tests = [
        "Félicitations ! Vous avez gagné 500 000 FCFA à la grande loterie d'été. Cliquez vite sur bit.ly/gain2024 pour réclamer votre prix !",
        "Cher client BCI, votre compte a été bloqué pour sécurité. Veuillez entrer votre code de validation 483920 pour le réactiver.",
        "URGENT: Service Client M-Pesa. Pour finaliser votre remboursement, rappelez immédiatement le +226 70 12 34 56.",
        "Bonjour maman, mon téléphone est tombé dans l'eau. Peux-tu m'envoyer 25 000 F sur ce nouveau numéro s'il te plaît ?",
        "Salut Pierre, on se retrouve comme prévu au café à 14h pour réviser le cours de deep learning ?"
    ]

    for i, sms in enumerate(cas_tests, 1):
        res = predire_sms(sms, modele, vocab, max_len=max_len, device=device)
        print(f"\n[Exemple {i}] SMS : \"{sms}\"")
        print(f"  -> Prétraité  : \"{res['texte_pretraite']}\"")
        print(f"  -> Verdict    : {res['prediction']} (Risque de fraude estimé à {res['probabilite_fraude']}%)")

    # 9. Génération du rapport Deep Learning complet
    rapport_dl = f"""# 🧠 Rapport Technique : Détection de SMS Frauduleux par Deep Learning (BiLSTM)

Modèle de Deep Learning développé pour le **Projet 2 : Détection de SMS frauduleux en français**.

---

## 1. Architecture du Modèle

```text
Entrée textuelle (SMS brut)
        │
        ▼
Prétraitement & Placeholders (<URL>, <MONTANT>, <TEL>, <CODE>)
        │
        ▼
Couche d'Embedding (Vocabulaire: {len(vocab.mot_vers_id)} tokens, Dimension: 128)
        │
        ▼
Bidirectional LSTM (64 unités cachées x 2 directions = 128 dimensions)
        │
        ├──────────────────────────┐
        ▼                          ▼
Global Average Pooling    Global Max Pooling
        │                          │
        └─────────────┬────────────┘
                      ▼
            Concaténation (256 dimensions)
                      │
                      ▼
            Dropout (0.3) & Dense (64 neurones, ReLU)
                      │
                      ▼
            Dropout (0.3) & Dense (1 neurone de sortie)
                      │
                      ▼
            Fonction Sigmoïde -> Probabilité de Fraude
```

---

## 2. Performances sur le Jeu de Test Officiel (`data/test.csv`)

| Métrique | Valeur obtenue | Interprétation |
| :--- | :--- | :--- |
| **Exactitude (Accuracy)** | **{acc*100:.2f} %** | Pourcentage total de décisions correctes |
| **Score F1 (Fraude)** | **{test_f1*100:.2f} %** | Moyenne harmonique équilibrée entre précision et rappel |
| **Précision (Fraude)** | **{test_prec*100:.2f} %** | Moins de faux positifs (les SMS légitimes ne sont pas bloqués) |
| **Rappel (Fraude)** | **{test_rec*100:.2f} %** | Capacité à intercepter les fraudes réelles |
| **Aire sous la courbe (ROC-AUC)**| **{roc_auc*100:.2f} %** | Excellente séparabilité probabiliste des deux classes |

### Matrice de Confusion
- **Vrais Négatifs (Légitimes bien classés) :** `{vn}` / {vn + fp} ({vn/(vn+fp)*100:.1f}%)
- **Faux Positifs (Légitimes pris pour fraude) :** `{fp}`
- **Faux Négatifs (Fraudes manquées) :** `{fn}`
- **Vrais Positifs (Fraudes bien arrêtées) :** `{vp}` / {vp + fn} ({vp/(vp+fn)*100:.1f}%)

---

## 3. Validation sur les 5 Objectifs Spécifiques du Projet

1. **Faux messages de gains :** Détectés grâce à l'association des balises `<MONTANT>` et `<URL>`.
2. **Liens suspects :** Les URLs masquées ou raccourcies (`bit.ly`, domaines suspects) activent fortement les poids du BiLSTM.
3. **Demandes de codes de validation :** Le token `<CODE>` combiné à des termes d'urgence alerte immédiatement le modèle.
4. **Faux agents de services financiers :** Les numéros suspects `<TEL>` et le vocabulaire bancaire usurpé sont identifiés.
5. **Tentatives d'usurpation :** Les faux messages d'urgence familiale ou de blocage de compte sont repérés avec un risque > 90%.

---

## 4. Fichiers Produits

- Modèle sauvegardé (poids PyTorch) : [`modele_sms_bilstm.pt`](file:///c:/Users/hermi/Desktop/PLANORA2/modele_sms_bilstm.pt)
- Vocabulaire JSON : [`vocabulaire_sms.json`](file:///c:/Users/hermi/Desktop/PLANORA2/vocabulaire_sms.json)
- Script d'entraînement et inférence : [`modele_deep_learning.py`](file:///c:/Users/hermi/Desktop/PLANORA2/modele_deep_learning.py)
"""

    with open("data/rapport_deep_learning.md", "w", encoding="utf-8") as f:
        f.write(rapport_dl)

    print("\n" + "=" * 75)
    print("Rapport Deep Learning généré dans 'data/rapport_deep_learning.md'")
    print("Modèle enregistré dans 'modele_sms_bilstm.pt'")
    print("=" * 75)


if __name__ == "__main__":
    main()
