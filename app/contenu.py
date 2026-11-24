"""Les textes de l'interface : conseils et exemples de SMS."""

# Chaque conseil = (nom de l'icône, texte)
CONSEILS_SPAM = [
    ("link_off", "Ne clique sur <b>aucun lien</b> contenu dans ce message."),
    ("phone_disabled", "Ne rappelle <b>jamais</b> le numéro indiqué : il peut être surtaxé."),
    ("lock", "Ne donne <b>jamais</b> ton code bancaire ou tes mots de passe par SMS."),
    ("flag", "Signale-le gratuitement en le transférant au <b>33700</b>, puis supprime-le."),
]

CONSEILS_LEGITIME = [
    ("thumb_up", "Ce message semble normal."),
    ("visibility", "Reste quand même vigilant si on te demande de l'argent ou des informations personnelles."),
    ("help", "En cas de doute, contacte l'expéditeur par un autre moyen (appel, en personne)."),
]

# Chaque exemple = (texte du bouton, icône, SMS)
EXEMPLES = [
    ("Exemple spam", ":material/report:",
     "Félicitations, vous avez gagné un bon d'achat de 500 euros. Cliquez ici pour le récupérer."),
    ("Exemple normal", ":material/chat:",
     "Votre rendez-vous est confirmé pour demain à 10h."),
]

# Les signes d'un spam, affichés dans la barre latérale
SIGNES_SPAM = """
:material/schedule: **L'urgence** : « agis maintenant »

:material/redeem: **Un cadeau trop beau** : iPhone gratuit

:material/link: **Un lien bizarre** ou raccourci

:material/credit_card: **Une demande d'infos** : code, RIB

:material/text_fields: **Des MAJUSCULES** partout
"""
