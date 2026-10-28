# Rapport de projet — SmartCar Ops (LaTeX)

Compilation avec Docker (aucune installation LaTeX sur la machine), depuis ce dossier :

```bash
docker run --rm -v "$PWD:/doc" -w /doc texlive/texlive:latest latexmk -pdf -interaction=nonstopmode main.tex
```

Sous PowerShell, remplacer `"$PWD:/doc"` par `"${PWD}:/doc"`. Le résultat est `main.pdf`.

## Organisation

| Fichier / dossier | Contenu |
|---|---|
| `main.tex` | préambule, mise en page, ordre des chapitres |
| `chapitres/` | page de garde, résumé, abréviations, introduction, chapitres 1 à 4, conclusion |
| `biblio.bib` | bibliographie (biblatex + biber) |
| `figures/photos/` | photos des composants et du prototype |
| `figures/captures/` | captures de l'application SmartCar Ops |
| `figures/ia/` | figures des notebooks d'entraînement (courbes, matrices, prédictions) |
| `figures/donnees/` | images des données du prototype (SAM 2, augmentations, piste) |
| `figures/tikz_*.tex` | diagrammes (architecture, flux de données, interactions) |

Le titre du projet se change en un seul endroit : la commande `\titreprojet` dans `main.tex`.
