# PFC_QA project notes

Power BI Desktop detecte:

```text
C:\Program Files\WindowsApps\Microsoft.MicrosoftPowerBIDesktop_2.155.756.0_x64__8wekyb3d8bbwe
```

Le fichier projet utilisateur est:

```text
powerbi\PFC_QA.pbip
```

Le projet est utilisable depuis Codex pour:

- generer les donnees source dans `powerbi\data`
- maintenir les requetes Power Query `.pq`
- maintenir les mesures DAX
- documenter le layout attendu

Les scripts PowerShell detectent la racine du repo depuis leur propre dossier.
Les requetes Power Query standalone passent par `queries/RepoRoot.pq`; le
modele TMDL est regenere avec le chemin local au moment du build.

L'edition visuelle finale reste a faire dans Power BI Desktop, car le format
des pages du rapport est fortement lie a l'application Desktop.
