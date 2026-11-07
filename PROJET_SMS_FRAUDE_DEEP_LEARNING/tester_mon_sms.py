"""
=============================================================================
Script d'inférence interactif : tester_mon_sms.py
Utilisation : python tester_mon_sms.py "Votre SMS à tester ici"
=============================================================================
"""

import sys
import io
import torch
from preparer_donnees import pretraiter
from modele_deep_learning import VocabulaireSMS, BiLSTM_SMS_Classifier, predire_sms

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def main():
    if len(sys.argv) < 2:
        sms_test = "URGENT ! Vous avez gagné 1 000 000 FCFA. Cliquez sur bit.ly/gain2024"
        print(f"Aucun SMS fourni en argument. Test avec le message d'exemple :\n\"{sms_test}\"\n")
    else:
        sms_test = " ".join(sys.argv[1:])

    # 1. Charger le vocabulaire
    vocab = VocabulaireSMS()
    vocab.charger("vocabulaire_sms.json")

    # 2. Charger l'architecture et les poids du modèle
    modele = BiLSTM_SMS_Classifier(
        vocab_size=len(vocab.mot_vers_id),
        embedding_dim=128,
        hidden_dim=64,
        num_layers=1,
        dropout=0.3
    )
    modele.load_state_dict(torch.load("modele_sms_bilstm.pt", map_location="cpu"))

    # 3. Prédiction
    res = predire_sms(sms_test, modele, vocab, max_len=60, device="cpu")

    print("=" * 65)
    print("RÉSULTAT DE L'ANALYSE DEEP LEARNING :")
    print("=" * 65)
    print(f"Texte brut       : {res['texte_original']}")
    print(f"Texte prétraité  : {res['texte_pretraite']}")
    print(f"Score de risque  : {res['probabilite_fraude']} %")
    print(f"Verdict final    : {res['prediction']}")
    print("=" * 65)

if __name__ == "__main__":
    main()
