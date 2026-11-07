import torch
from modele_deep_learning import (
    VocabulaireSMS,
    BiLSTM_SMS_Classifier,
    predire_sms,
)

# Classe attendue : 0 = légitime, 1 = fraude.
# Ces messages sont synthétiques et servent à explorer des sujets variés.
cas = [
    ("Santé", 0, "Rappel : votre rendez-vous au centre médical est confirmé pour jeudi à 14 h."),
    ("Santé", 1, "Votre remboursement santé est prêt. Pour le recevoir, installez notre application et saisissez vos identifiants."),
    ("Études", 0, "La salle du cours de biologie a changé : rendez-vous en B204 à 9 h."),
    ("Études", 1, "Bourse étudiante exceptionnelle : envoyez vos coordonnées bancaires pour valider votre dossier aujourd'hui."),
    ("Travail", 0, "La réunion de l'équipe commence à 10 h demain. Pensez à apporter le compte rendu."),
    ("Travail", 1, "Votre candidature est retenue. Payez les frais de dossier avant l'entretien pour confirmer le poste."),
    ("Transport", 0, "Perturbation sur la ligne de bus : le prochain départ aura environ 15 minutes de retard."),
    ("Transport", 1, "Amende de stationnement impayée. Scannez le QR code reçu pour éviter une majoration."),
    ("Logement", 0, "Le technicien passera mardi entre 13 h et 15 h pour vérifier le chauffage."),
    ("Logement", 1, "Le propriétaire a changé de compte. Versez la caution sur ce nouveau compte avant la remise des clés."),
    ("Énergie", 0, "Votre relevé d'électricité est disponible dans l'application habituelle. Aucune action urgente n'est demandée."),
    ("Énergie", 1, "Coupure prévue ce soir. Répondez avec le code reçu par SMS pour maintenir votre contrat."),
    ("Livraison", 0, "Votre colis sera livré demain entre 8 h et 12 h. Vous n'avez rien à payer."),
    ("Livraison", 1, "Adresse incomplète : payez les frais de nouvelle livraison sur https://suivi-colis.example/maj"),
    ("Famille", 0, "Je serai en retard au dîner, garde-moi une place près de la fenêtre !"),
    ("Famille", 1, "C'est moi, j'ai perdu mon téléphone. Achète trois cartes cadeaux et envoie-moi les codes, je t'explique après."),
    ("Compte en ligne", 0, "Vous avez demandé la réinitialisation de votre mot de passe. Si ce n'était pas vous, ignorez ce message."),
    ("Compte en ligne", 1, "Assistance technique : installez l'outil de contrôle à distance et laissez votre téléphone déverrouillé."),
    ("Style abrégé", 0, "Slt, le train a 10 min de retard, je t'attends devant la gare."),
    ("Style abrégé", 1, "Dernier rappel!!! votre dossier est bloqué, envoyez vos infos mtn sinon tout sera supprimé"),
]

vocab = VocabulaireSMS()
vocab.charger("vocabulaire_sms.json")

modele = BiLSTM_SMS_Classifier(
    vocab_size=len(vocab.mot_vers_id),
    embedding_dim=128,
    hidden_dim=64,
    num_layers=1,
    dropout=0.3,
)
modele.load_state_dict(torch.load("modele_sms_bilstm.pt", map_location="cpu"))

total_correct = 0

for theme, attendu, sms in cas:
    resultat = predire_sms(sms, modele, vocab, max_len=60, device="cpu")
    predit = resultat["est_fraude"]
    total_correct += predit == attendu

    nom_attendu = "FRAUDE" if attendu else "LÉGITIME"
    nom_predit = "FRAUDE" if predit else "LÉGITIME"
    comparaison = "OK" if predit == attendu else "À EXAMINER"

    print(f"\n[{theme}] {comparaison}")
    print(f"  Attendu : {nom_attendu} | Prédit : {nom_predit}")
    print(f"  Risque estimé : {resultat['probabilite_fraude']} %")
    print(f"  SMS : {sms}")

print(f"\nAccord sur ces exemples synthétiques : {total_correct}/{len(cas)}")

cas_ambigus = [
    ("Livraison", "Votre colis est en attente. Des frais de 1,20 € peuvent s'appliquer pour une nouvelle présentation."),
    ("Banque", "Une opération inhabituelle a été détectée. Vérifiez votre activité dans l'application habituelle."),
    ("Santé", "Votre document médical est disponible. Connectez-vous à votre espace patient pour le consulter."),
    ("Travail", "Nous avons bien reçu votre candidature. Pour poursuivre, merci de compléter votre profil en ligne."),
    ("Famille", "Tu peux m'envoyer un peu d'argent ? Je te rembourse dès que je rentre."),
    ("Énergie", "Une régularisation figure sur votre prochaine facture. Consultez le détail dans votre espace client."),
    ("Concours", "Vous avez été sélectionné pour recevoir un cadeau. Confirmez votre adresse avant la fin de la journée."),
    ("Compte en ligne", "Une connexion depuis un nouvel appareil a été détectée. Si ce n'est pas vous, sécurisez votre compte."),
    ("Association", "Notre collecte se termine ce soir. Chaque don, même petit, nous aide à poursuivre le projet."),
    ("Transport", "Votre abonnement arrive à expiration. Renouvelez-le avant lundi pour éviter une interruption."),
]

print("\n--- CAS AMBIGUS : À ANALYSER SANS CLASSE ATTENDUE ---")

for theme, sms in cas_ambigus:
    resultat = predire_sms(sms, modele, vocab, max_len=60, device="cpu")
    print(f"\n[{theme}]")
    print(f"  Risque estimé : {resultat['probabilite_fraude']} %")
    print(f"  Verdict : {resultat['prediction']}")
    print(f"  SMS : {sms}")