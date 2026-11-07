"""Interface graphique pour analyser un SMS avec le modèle existant."""

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import torch

from modele_deep_learning import (
    BiLSTM_SMS_Classifier,
    VocabulaireSMS,
    predire_sms,
)


DOSSIER_PROJET = Path(__file__).resolve().parent
COULEUR_FOND = "#f6f3ff"
COULEUR_VIOLET = "#6734d2"
COULEUR_VIOLET_FONCE = "#4820a4"
COULEUR_TEXTE = "#1d1745"
COULEUR_SECONDAIRE = "#655c86"
PLACEHOLDER = "Exemple : Bonjour, votre rendez-vous est confirmé pour demain à 10 h."


class InterfaceSMS:
    def __init__(self, fenetre):
        self.fenetre = fenetre
        self.fenetre.title("SMS Sentinelle - Détection de SMS frauduleux")
        self.fenetre.geometry("960x800")
        self.fenetre.minsize(700, 680)
        self.fenetre.configure(bg=COULEUR_FOND)

        self._charger_modele()
        self._creer_interface()

    def _charger_modele(self):
        chemin_vocabulaire = DOSSIER_PROJET / "vocabulaire_sms.json"
        chemin_modele = DOSSIER_PROJET / "modele_sms_bilstm.pt"

        for chemin in (chemin_vocabulaire, chemin_modele):
            if not chemin.is_file():
                raise FileNotFoundError(f"Fichier nécessaire introuvable : {chemin}")

        self.vocab = VocabulaireSMS()
        self.vocab.charger(str(chemin_vocabulaire))

        self.modele = BiLSTM_SMS_Classifier(
            vocab_size=len(self.vocab.mot_vers_id),
            embedding_dim=128,
            hidden_dim=64,
            num_layers=1,
            dropout=0.3,
        )
        poids = torch.load(str(chemin_modele), map_location="cpu")
        self.modele.load_state_dict(poids)
        self.modele.eval()

    def _creer_interface(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Fraude.Horizontal.TProgressbar",
            troughcolor="#eadfff",
            background="#e34b59",
            thickness=12,
            borderwidth=0,
        )
        style.configure(
            "Legitime.Horizontal.TProgressbar",
            troughcolor="#e7e1f4",
            background="#22a778",
            thickness=12,
            borderwidth=0,
        )

        conteneur = tk.Frame(self.fenetre, bg=COULEUR_FOND, padx=42, pady=28)
        conteneur.pack(fill="both", expand=True)

        entete = tk.Frame(conteneur, bg=COULEUR_FOND)
        entete.pack(fill="x", pady=(0, 20))

        icone = tk.Canvas(
            entete, width=76, height=76, bg=COULEUR_FOND, highlightthickness=0
        )
        icone.pack(side="left", padx=(0, 16))
        icone.create_polygon(
            38, 4, 67, 15, 65, 43, 53, 61, 38, 71, 23, 61, 11, 43, 9, 15,
            fill=COULEUR_VIOLET, outline="#4c23ad", width=2,
        )
        icone.create_oval(21, 22, 53, 48, fill="white", outline="")
        icone.create_polygon(26, 42, 25, 53, 36, 46, fill="white", outline="")
        for x in (30, 37, 44):
            icone.create_oval(x - 1.7, 33, x + 1.7, 36.5, fill=COULEUR_VIOLET, outline="")

        titre = tk.Frame(entete, bg=COULEUR_FOND)
        titre.pack(side="left", fill="y", expand=True, anchor="w")
        ligne_titre = tk.Frame(titre, bg=COULEUR_FOND)
        ligne_titre.pack(anchor="w", pady=(5, 0))
        tk.Label(
            ligne_titre, text="SMS ", font=("Segoe UI", 30, "bold"),
            fg=COULEUR_TEXTE, bg=COULEUR_FOND,
        ).pack(side="left")
        tk.Label(
            ligne_titre, text="Sentinelle", font=("Segoe UI", 30, "bold"),
            fg=COULEUR_VIOLET, bg=COULEUR_FOND,
        ).pack(side="left")
        tk.Label(
            titre,
            text="Détectez les SMS frauduleux en un instant",
            font=("Segoe UI", 12),
            fg=COULEUR_SECONDAIRE,
            bg=COULEUR_FOND,
        ).pack(anchor="w", pady=(1, 0))
        tk.Label(
            entete,
            text="PROTECTION SMS",
            font=("Segoe UI", 9, "bold"),
            fg=COULEUR_VIOLET_FONCE,
            bg="#ece5ff",
            padx=12,
            pady=7,
        ).pack(side="right", anchor="n", pady=(10, 0))

        carte = tk.Frame(
            conteneur,
            bg="white",
            padx=28,
            pady=24,
            highlightbackground="#e2d9f8",
            highlightthickness=1,
        )
        carte.pack(fill="both", expand=True)

        titre_saisie = tk.Frame(carte, bg="white")
        titre_saisie.pack(fill="x")
        tk.Label(
            titre_saisie,
            text="●",
            font=("Segoe UI", 18, "bold"),
            fg=COULEUR_VIOLET,
            bg="white",
        ).pack(side="left", padx=(0, 10))
        tk.Label(
            titre_saisie,
            text="Entrez votre SMS",
            font=("Segoe UI", 17, "bold"),
            fg=COULEUR_TEXTE,
            bg="white",
        ).pack(side="left")
        tk.Label(
            carte,
            text="Saisissez ou collez le message que vous souhaitez analyser.",
            font=("Segoe UI", 10),
            fg=COULEUR_SECONDAIRE,
            bg="white",
        ).pack(anchor="w", pady=(3, 12))

        zone_saisie = tk.Frame(
            carte,
            bg="white",
            highlightbackground="#bca7f6",
            highlightthickness=1,
        )
        zone_saisie.pack(fill="both", expand=True)
        barre = tk.Scrollbar(zone_saisie, relief="flat")
        barre.pack(side="right", fill="y")
        self.saisie = tk.Text(
            zone_saisie,
            height=5,
            wrap="word",
            font=("Segoe UI", 12),
            fg="#82799f",
            bg="white",
            relief="flat",
            padx=14,
            pady=13,
            yscrollcommand=barre.set,
            undo=True,
        )
        self.saisie.pack(fill="both", expand=True)
        barre.configure(command=self.saisie.yview)
        self.saisie.insert("1.0", PLACEHOLDER)
        self.saisie.bind("<FocusIn>", self._retirer_placeholder)
        self.saisie.bind("<FocusOut>", self._afficher_placeholder)

        commandes = tk.Frame(carte, bg="white")
        commandes.pack(fill="x", pady=15)
        self.bouton_analyser = tk.Button(
            commandes,
            text="⌕   Analyser le SMS",
            command=self._analyser,
            font=("Segoe UI", 14, "bold"),
            fg="white",
            bg=COULEUR_VIOLET,
            activeforeground="white",
            activebackground=COULEUR_VIOLET_FONCE,
            relief="flat",
            cursor="hand2",
            pady=12,
            bd=0,
        )
        self.bouton_analyser.pack(side="left", fill="x", expand=True)
        self.bouton_analyser.bind(
            "<Enter>", lambda _event: self.bouton_analyser.configure(bg="#5828bc")
        )
        self.bouton_analyser.bind(
            "<Leave>", lambda _event: self.bouton_analyser.configure(bg=COULEUR_VIOLET)
        )
        tk.Button(
            commandes,
            text="Effacer",
            command=self._effacer,
            font=("Segoe UI", 11, "bold"),
            fg=COULEUR_TEXTE,
            bg="#f0ebfb",
            activebackground="#e4daf8",
            relief="flat",
            cursor="hand2",
            padx=18,
            pady=12,
            bd=0,
        ).pack(side="left", padx=(10, 0))

        self.carte_resultat = tk.Frame(
            carte,
            bg="#f7f3ff",
            padx=22,
            pady=17,
            highlightbackground="#ddd0ff",
            highlightthickness=1,
        )
        self.carte_resultat.pack(fill="x")
        tk.Label(
            self.carte_resultat,
            text="RÉSULTAT DE L'ANALYSE",
            font=("Segoe UI", 9, "bold"),
            fg="#77699c",
            bg="#f7f3ff",
        ).pack(anchor="center")

        self.verdict = tk.Label(
            self.carte_resultat,
            text="En attente d'une analyse",
            font=("Segoe UI", 17, "bold"),
            fg=COULEUR_TEXTE,
            bg="#f7f3ff",
        )
        self.verdict.pack(anchor="center", pady=(5, 0))
        self.libelle_score = tk.Label(
            self.carte_resultat,
            text="Risque de fraude estimé",
            font=("Segoe UI", 12, "bold"),
            fg="#3b315f",
            bg="#f7f3ff",
        )
        self.libelle_score.pack(anchor="center", pady=(5, 0))
        self.score = tk.Label(
            self.carte_resultat,
            text="-- %",
            font=("Segoe UI", 30, "bold"),
            fg=COULEUR_VIOLET,
            bg="#f7f3ff",
        )
        self.score.pack(anchor="center", pady=(0, 4))
        self.barre_risque = ttk.Progressbar(
            self.carte_resultat,
            mode="determinate",
            maximum=100,
            style="Legitime.Horizontal.TProgressbar",
        )
        self.barre_risque.pack(fill="x", pady=(0, 10))
        self.texte_pretraite = tk.Label(
            self.carte_resultat,
            text="Le résultat de votre analyse apparaîtra ici.",
            font=("Segoe UI", 10),
            fg=COULEUR_SECONDAIRE,
            bg="#f7f3ff",
            wraplength=760,
            justify="center",
        )
        self.texte_pretraite.pack(anchor="center")

        tk.Label(
            carte,
            text=(
                "Cette analyse est une aide et peut se tromper. "
                "En cas de doute, contactez directement l'organisme concerné."
            ),
            font=("Segoe UI", 9),
            fg="#766d8f",
            bg="white",
            wraplength=760,
            justify="center",
        ).pack(anchor="center", pady=(15, 0))

        self.fenetre.bind("<Control-Return>", lambda _event: self._analyser())

    def _retirer_placeholder(self, _event=None):
        if self.saisie.get("1.0", "end-1c") == PLACEHOLDER:
            self.saisie.delete("1.0", "end")
            self.saisie.configure(fg=COULEUR_TEXTE)

    def _afficher_placeholder(self, _event=None):
        if not self.saisie.get("1.0", "end-1c").strip():
            self.saisie.insert("1.0", PLACEHOLDER)
            self.saisie.configure(fg="#82799f")

    def _analyser(self):
        sms = self.saisie.get("1.0", "end").strip()
        if not sms or sms == PLACEHOLDER:
            messagebox.showwarning(
                "SMS manquant",
                "Veuillez saisir ou coller un SMS avant de lancer l'analyse.",
                parent=self.fenetre,
            )
            return

        resultat = predire_sms(sms, self.modele, self.vocab, max_len=60, device="cpu")
        proba = float(resultat["probabilite_fraude"])

        if resultat["est_fraude"]:
            couleur = "#c83243"
            fond_resultat = "#fff3f3"
            self.verdict.configure(text="⚠  SMS potentiellement frauduleux", fg=couleur)
            self.barre_risque.configure(style="Fraude.Horizontal.TProgressbar")
        else:
            couleur = "#16845e"
            fond_resultat = "#effaf5"
            self.verdict.configure(text="✓  SMS probablement légitime", fg=couleur)
            self.barre_risque.configure(style="Legitime.Horizontal.TProgressbar")

        self.carte_resultat.configure(
            bg=fond_resultat, highlightbackground="#f0c4ca" if resultat["est_fraude"] else "#bde8d4"
        )
        for widget in (self.verdict, self.score, self.libelle_score, self.texte_pretraite):
            widget.configure(bg=fond_resultat)
        self.score.configure(text=f"{proba:.1f} %", fg=couleur)
        self.libelle_score.configure(text="Risque de fraude estimé")
        self.barre_risque["value"] = proba
        self.texte_pretraite.configure(
            text=f"Texte après prétraitement : {resultat['texte_pretraite']}"
        )

    def _effacer(self):
        self.saisie.delete("1.0", "end")
        self.saisie.configure(fg=COULEUR_TEXTE)
        self.verdict.configure(text="En attente d'une analyse", fg=COULEUR_TEXTE)
        self.score.configure(text="-- %", fg=COULEUR_VIOLET)
        self.libelle_score.configure(text="Risque de fraude estimé")
        self.carte_resultat.configure(bg="#f7f3ff", highlightbackground="#ddd0ff")
        for widget in (self.verdict, self.score, self.libelle_score, self.texte_pretraite):
            widget.configure(bg="#f7f3ff")
        self.barre_risque["value"] = 0
        self.barre_risque.configure(style="Legitime.Horizontal.TProgressbar")
        self.texte_pretraite.configure(text="Le résultat de votre analyse apparaîtra ici.")
        self.saisie.focus_set()


def main():
    fenetre = tk.Tk()
    fenetre.withdraw()

    try:
        application = InterfaceSMS(fenetre)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as erreur:
        messagebox.showerror(
            "Impossible de charger le modèle",
            f"{erreur}\n\nVérifiez la présence du modèle et du vocabulaire dans le dossier du projet.",
            parent=fenetre,
        )
        fenetre.destroy()
        raise

    fenetre.deiconify()
    fenetre.mainloop()


if __name__ == "__main__":
    main()
