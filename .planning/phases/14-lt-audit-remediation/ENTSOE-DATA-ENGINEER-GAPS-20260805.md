# ENTSO-E - points à fermer côté data engineering

État au 2026-08-05. Aucun accès Databricks n'a été déclenché pour ce contrôle.

## Verdict court

Les groupes utiles existent dans `dev.gold`, mais ils ne sont pas encore
admissibles pour entraîner ou sélectionner la PFC LT. Les données locales
peuvent servir au développement du schéma et des contrôles uniquement.

## À livrer

1. **Export long et non ambigu**
   - dimension au grain `series_id` ;
   - latest au grain `series_id, target_ts_utc` ;
   - vintages au grain `series_id, target_ts_utc, as_of_utc`.

2. **Unités et cadence natives**
   - `MW`, `MWh` et prix explicitement `EUR/MWh`, jamais seulement `EUR` ;
   - résolution courante par série et historique effectif des régimes au
     grain `series_id, valid_from_utc`, avec fin, source et confirmation ;
   - aucun upsampling ou forward-fill implicite.

3. **Vraie traçabilité PIT**
   - `as_of_utc`, révision, qualité, timestamp de chargement, document et
     endpoint source ;
   - les backfills historiques doivent être signalés et ne doivent jamais être
     présentés comme disponibles aux anciennes dates de calcul.

4. **Couverture par série**
   - CH load/génération, solaire, éolien, nucléaire et hydro ;
   - flux physiques, échanges programmés et NTC pour CH-DE/FR/IT/AT ;
   - pour chaque famille/frontière, identifier séparément CH vers voisin et
     voisin vers CH ; une seule série ne couvre les deux sens que si elle est
     réellement signée et que son sens positif est documenté ;
   - forecasts et stockage hydro ;
   - profil des trous contre la cadence native. Pour un diagnostic saisonnier
     cross-market, viser au minimum 730 jours complets avant toute étude de
     sélection ; le rolling-origin pourra exiger davantage.
   - fournir le registre temporel des zones `CH`, `DE_LU`, `FR`, `IT_NORTH`,
     `AT` : identifiant source exact, schéma d'identifiant, type de domaine,
     `valid_from_utc`, `valid_to_utc`, statut, type/ID du document de registre
     et endpoint HTTPS. `DocumentType` n'est pas un identifiant de registre.

5. **Une seule vérité canonique**
   - expliquer et corriger la divergence entre les vues combinée et dédiée des
     fondamentaux ; l'import local actuel diverge sur 11 séries ;
   - garantir les clés uniques et 100 % de jointure fact-dimension.

6. **Snapshot immuable**
   - export unique, manifeste avec tailles, nombres de lignes et SHA-256 ;
   - aucune écriture depuis le pipeline PFC vers Databricks.

Le contrat machine complet est
`ENTSOE-DATABRICKS-INTAKE-CONTRACT-V1.json`. Jusqu'à son passage complet, statut
strict : `NO_GO` pour entraînement, rolling-origin, sélection et promotion.

## Mise à jour du 2026-08-06 sans requête SQL

Les API de contrôle confirment les trois tables Delta managées, mais leur schéma
réel manque encore de `native_resolution`, `source_endpoint`, identifiant de
document source, `quality_flag`, `revision_number` et convention de signe. Les
vintages exposent `PublicationTimestampUtc` et `PullTimestampUtc` : le data
engineer doit désigner la règle canonique de construction de `as_of_utc`.

L'inventaire exact des `GroupName`/`FieldName` n'a pas été lancé : les deux
Warehouses visibles sont arrêtés, classiques `2X-Small`, avec auto-stop à
45 minutes. Voir `ENTSOE-DEV-CONTROL-PLANE-INVENTORY-20260806.md` pour les
colonnes réelles et les familles supplémentaires recommandées.

Le contrôle automatisé D241 confirme exactement 11/8/10 colonnes et refuse
l'autorité modèle. Il faut donc fournir les champs manquants ci-dessus ou une
règle de transformation documentée, testable et sans fabrication. Le choix
entre `PublicationTimestampUtc` et `PullTimestampUtc` pour `as_of_utc` doit être
explicite ; `latest` ne peut pas servir aux rolling origins historiques.

## Ce que montre déjà `dev.gold` au 2026-08-03

- groupes CH utiles présents et données latest récentes ;
- premières vraies vintages observées seulement le 2026-07-28 ;
- historique antérieur principalement backfillé, donc pas une reconstitution
  de l'information disponible aux anciennes origines ;
- certains flux présentent encore des trous longs à attribuer par série ;
- unité métier des prix confirmée en `EUR/MWh` ; le contrat de données doit
  conserver ce libellé explicite et sa traçabilité de source ;
- la cadence des prix doit suivre le régime de marché suisse applicable : un
  historique horaire n'est pas un manque de vérité 15 minutes. Aucun
  upsampling implicite ne doit être utilisé pour créer une fausse granularité.

## Statut de couverture consolidé au 2026-08-06

Les macro-familles charge réelle, prix day-ahead, production réelle/forecast,
forecasts solaire/éolien, stockage réservoir, flux physiques, échanges
programmés et NTC day/month/year ont été observées le 2026-08-03. Restent à
prouver explicitement : load forecast, séparation renouvelables day-ahead vs
intraday, détail complet des technologies, quatre frontières CH dans les deux
sens utiles, résolution native, `EUR/MWh`, signes, document source, qualité,
révision et règle PIT.

À ajouter si absent : outages production/réseau, capacité installée/disponible,
balancing aFRR/mFRR/RR prix+volumes up/down, imbalance price/system imbalance,
réserves capacité/prix, redispatch/countertrading, net positions et capacité
cross-zonale intraday.

Second niveau utile, non bloquant pour la première baseline : production
réelle par unité, curtailment, marge de forecast de charge, paramètres de
capacité flow-based et capacité cross-zonale de balancing allouée/utilisée.

La tentative réservée du 2026-08-06 n'a lancé aucun SQL : Warehouse `STOPPED`,
classique `2X-Small`, auto-stop 45 minutes. La journée est consommée et aucun
retry n'est autorisé.

## Correction D250 à appliquer au mapping réel

Les commentaires du schéma réel indiquent que `DateTimeUtc` est la **borne
droite** de l'intervalle. La PFC attend un **début d'intervalle**. Il ne faut
donc pas renommer directement la colonne : exporter
`target_ts_utc = DateTimeUtc - native_resolution`.

Demande courte au data engineer :

1. publier la résolution courante et l'historique effectif des régimes par
   `SeriesID`, sans supposer partout 1 heure ou 15 minutes ;
2. confirmer la convention de borne droite et appliquer la transformation
   ci-dessus ;
3. conserver publication, pull et load séparément, puis faire approuver la
   règle PIT proposée `max` avec rejet des nulls ;
4. publier l'identifiant immuable du document source, l'endpoint, la qualité,
   la révision et la convention de signe ;
5. fournir l'inventaire exact `GroupName` / `FieldName` / unité / frontière et
   signaler explicitement toute ligne rejetée ou perte de couverture.

La transformation temporelle doit utiliser la résolution active de chaque
intervalle : `target_ts_utc = DateTimeUtc - active_native_resolution`. Une
date de roadmap suisse ne remplace jamais la preuve propre à la série.

`DocumentType` n'est pas un identifiant de document, `VintageID` n'est pas une
révision source, et aucune valeur `OK`, révision zéro, cadence ou signe ne doit
être inventé. `latest` reste interdit pour reconstituer les historiques PIT.
Le message autonome à transmettre est dans
`ENTSOE-REAL-MAPPING-DATA-ENGINEER-REQUEST-20260806.md`.

## Ajout D255 : rattachement exact des séries aux zones

Merci de livrer aussi :

1. l'identifiant brut exact présent dans `FromZone` ou `ToZone` pour chaque
   série non directionnelle concernant `CH`, `DE_LU`, `FR`, `IT_NORTH` et
   `AT` ;
2. le code EIC ou libellé gouverné correspondant, avec `valid_from_utc` et
   `valid_to_utc` ;
3. une ligne distincte à chaque changement de code ou de configuration ;
4. la correspondance de chaque série avec son entrée exacte du registre.

Les gabarits sans valeur sont dans `docs/data/templates/`. Ils ne doivent être
complétés qu'après l'inventaire D243. Les flux, échanges programmés et NTC
restent traités séparément par frontière et par sens.

Ce contrôle confirme la structure attendue, pas la complétude réelle de
`dev.gold`. À ce stade, les points encore à prouver sont surtout : load
forecast, horizons day-ahead/intraday, détail des technologies, cinq zones de
prix, quatre frontières CH dans les deux sens, cadence par régime, unité
`EUR/MWh`, signe, qualité, révision, source et disponibilité PIT.
