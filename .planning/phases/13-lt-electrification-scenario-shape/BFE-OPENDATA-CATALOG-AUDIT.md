# BFE opendata.swiss Catalogue Audit

* organization: `bundesamt-fur-energie-bfe`
* datasets scanned: `146`
* csv: `data\bfe_opendata_catalog_audit.csv`

## Summary

| priority | family | datasets |
| --- | --- | --- |
| P0 | demand_buildings,energiedashboard | 1 |
| P0 | electricity_balance | 9 |
| P0 | electricity_balance,demand_buildings | 6 |
| P0 | electricity_balance,demand_buildings,energiedashboard | 1 |
| P0 | electricity_balance,energiedashboard | 4 |
| P0 | electricity_balance,ev_mobility | 1 |
| P0 | electricity_balance,pv | 3 |
| P0 | energiedashboard | 1 |
| P0 | hydro_reservoir | 5 |
| P0 | hydro_reservoir,electricity_balance | 1 |
| P0 | hydro_reservoir,pv,prices_market | 1 |
| P0 | installed_capacity | 4 |
| P0 | installed_capacity,electricity_balance,pv,energiedashboard | 1 |
| P0 | installed_capacity,pv | 2 |
| P0 | installed_capacity,pv,prices_market | 1 |
| P0 | prices_market,energiedashboard | 1 |
| P0 | pv | 6 |
| P0 | pv,demand_buildings | 4 |
| P0 | pv,ev_mobility,demand_buildings | 1 |
| P0 | pv,prices_market | 1 |
| P1 | demand_buildings | 25 |
| P1 | ev_mobility | 9 |
| P1 | ev_mobility,demand_buildings | 1 |
| P1 | prices_market | 4 |
| P2 | grid_infrastructure | 3 |
| P3 | other | 50 |

## P0/P1 Candidates

| priority | family | dataset_name | title_en | formats | first_resource_url |
| --- | --- | --- | --- | --- | --- |
| P0 | demand_buildings,energiedashboard | energiedashboard-ch-fullstande-gasspeicher-eu | energiedashboard.ch: Gas storage levels EU | CSV | https://www.bfe-ogd.ch/ogd102_fuellstand_gasspeicher.csv |
| P0 | electricity_balance | absatz-und-stromverbrauchswerte-von-elektro-und-elektronischen-gerate-in-der-schweiz |  | CSV | https://www.uvek-gis.admin.ch/BFE/ogd/109/ogd109_catalog_geraetekategorien.csv |
| P0 | electricity_balance | ruckerstattung-nach-art-15b-eng-fur-stromintensive-endverbraucher |  | SERVICE | https://pubdb.bfe.admin.ch/de/suche?keywords=397 |
| P0 | electricity_balance | schweizerische-elektrizitatsstatistik-aufteilung-des-endverbrauchs-nach-verbrauchergruppen |  | SERVICE | https://pubdb.bfe.admin.ch/de/suche?keywords=400 |
| P0 | electricity_balance | schweizerische-elektrizitatsstatistik-aussenhandel-der-schweiz |  | SERVICE | https://pubdb.bfe.admin.ch/de/suche?keywords=687 |
| P0 | electricity_balance | schweizerische-elektrizitatsstatistik-aussenhandel-der-schweiz-mit-elektrizitat-nach-landern |  | SERVICE | https://pubdb.bfe.admin.ch/de/suche?keywords=332 |
| P0 | electricity_balance | schweizerische-gesamtenergiestatistik-erdolbilanz-der-schweiz |  | CSV | https://www.uvek-gis.admin.ch/BFE/ogd/124/ogd124_erdölbilanz.csv |
| P0 | electricity_balance | stromlandschaft-schweiz |  | CSV | https://www.uvek-gis.admin.ch/BFE/ogd/85/ogd85_stromlandschaft.csv |
| P0 | electricity_balance | wettbewerbliche-ausschreibungen-fuer-stromeffizienzmassnahmen-gefoerderte-programme-und-projekte |  | SERVICE | https://pubdb.bfe.admin.ch/de/suche?keywords=396 |
| P0 | electricity_balance | wochenstatistik-elektrizitatsbilanz-erzeugung-und-abgabe-elektrischer-energie-in-der-schweiz |  | CSV,SERVICE | https://pubdb.bfe.admin.ch/de/suche?keywords=412 |
| P0 | electricity_balance,demand_buildings | kehrichtverbrennungsanlagen-kva | Waste incineration plants (MWI) | HTML,OCTET,XML | https://data.geo.admin.ch/ch.bfe.kehrichtverbrennungsanlagen/kehrichtverbrennungsanlagen/kehrichtverbrennungsanlagen_2056.csv.zip |
| P0 | electricity_balance,demand_buildings | schweizerische-gesamtenergiestatistik |  | SERVICE | https://pubdb.bfe.admin.ch/de/suche?keywords=402 |
| P0 | electricity_balance,demand_buildings | schweizerische-holzenergiestatistik |  | SERVICE | https://pubdb.bfe.admin.ch/de/suche?keywords=403 |
| P0 | electricity_balance,demand_buildings | schweizerische-statistik-der-erneuerbaren-energien |  | SERVICE | https://pubdb.bfe.admin.ch/de/suche?keywords=404 |
| P0 | electricity_balance,demand_buildings | teilstatistik-spezielle-energetische-holznutzungen-feuerungen-und-motoren-fur-erneuerbare-abfalle |  | SERVICE | https://pubdb.bfe.admin.ch/de/suche?keywords=408 |
| P0 | electricity_balance,demand_buildings | thermische-stromproduktion-inklusive-warmekraftkopplung-wkk-in-der-schweiz |  | SERVICE | https://pubdb.bfe.admin.ch/de/suche?keywords=409 |
| P0 | electricity_balance,demand_buildings,energiedashboard | energiedashboard-ch-tagliche-flusse-in-die-und-aus-der-schweiz-gas | energiedashboard.ch: Daily import and export flows (Gas) | CSV | https://www.bfe-ogd.ch/ogd101_gas_import_export.csv |
| P0 | electricity_balance,energiedashboard | energiedashboard-ch-landesverbrauch-und-endverbrauch | energiedashboard.ch - National and final consumption | CSV | https://www.bfe-ogd.ch/ogd103_stromverbrauch_swissgrid_lv_und_endv.csv |
| P0 | electricity_balance,energiedashboard | energiedashboard-ch-modellbasiert-geschatzter-landesverbrauch | energiedashboard.ch - Model-based estimated national consumption | CSV | https://www.bfe-ogd.ch/ogd103_stromverbrauch_geschaetzt_swissgrid.csv |
| P0 | electricity_balance,energiedashboard | energiedashboard-ch-monatliche-gas-nettoimporte-vsg | energiedashboard.ch: Monthly gas net imports VSG | CSV | https://www.uvek-gis.admin.ch/BFE/ogd/111/ogd111_gas_nettoimport.csv |
| P0 | electricity_balance,energiedashboard | energiedashboard-ch-stromverbrauch-prognose-sdsc | energiedashboard.ch - Power consumption forecast SDSC | CSV | https://www.bfe-ogd.ch/ogd110_strom_verbrauch_prognose.csv |
| P0 | electricity_balance,ev_mobility | ladebedarf-strombedarf-fur-steckerfahrzeuge-fur-das-jahr-2035 | Electricity requirements for plug-in vehicles for the year 2035 | HTML,OCTET,XML | https://data.geo.admin.ch/browser/index.html#/collections/ch.bfe.ladebedarfswelt/items/ladebedarfswelt |
| P0 | electricity_balance,pv | bezugerinnen-und-bezuger-der-einspeisevergutung-kev |  | CSV,SERVICE | https://pubdb.bfe.admin.ch/de/suche?keywords=383 |
| P0 | electricity_balance,pv | schweizerische-elektrizitatsbilanz-jahreswerte |  | CSV |  https://www.uvek-gis.admin.ch/BFE/ogd/32/ogd32_elektrizitaetbilanz_jahreswerte.csv |
| P0 | electricity_balance,pv | schweizerische-elektrizitatsstatistik-schweizerische-elektrizitatsbilanz-monatswerte |  | CSV,SERVICE | https://pubdb.bfe.admin.ch/de/suche?keywords=401 |
| P0 | energiedashboard | energiedashboard-ch-tagliche-flusse-in-die-und-aus-der-schweiz-strom | energiedashboard.ch: Daily import and export flows (Electricity) | CSV | https://www.bfe-ogd.ch/ogd107_strom_import_export.csv |
| P0 | hydro_reservoir | fullungsgrad-der-speicherseen-sonntag-24h-wochenbericht-speicherinhalt |  | CSV,SERVICE | https://www.uvek-gis.admin.ch/BFE/ogd/17/ogd17_fuellungsgrad_speicherseen.csv |
| P0 | hydro_reservoir | gesamte-erzeugung-und-abgabe-elektrischer-energie-in-der-schweiz |  | SERVICE | https://pubdb.bfe.admin.ch/de/suche?keywords=390 |
| P0 | hydro_reservoir | kleinwasserkraftpotentiale-der-schweizer-gewasser | Potential of small hydropower plants in Switzerland | HTML,OCTET,XML | https://data.geo.admin.ch/ch.bfe.kleinwasserkraftpotentiale/kleinwasserkraftpotentiale/kleinwasserkraftpotentiale_2056.gpkg.zip |
| P0 | hydro_reservoir | statistik-der-wasserkraftanlagen-wasta | Statistics on hydropower plants (WASTA) | HTML,OCTET,XML,ZIP | https://data.geo.admin.ch/ch.bfe.statistik-wasserkraftanlagen/statistik-wasserkraftanlagen/statistik-wasserkraftanlagen_2056.csv.zip |
| P0 | hydro_reservoir | stauanlagen-unter-bundesaufsicht | Dams under federal supervision | HTML,OCTET,XML | https://wms.geo.admin.ch/?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities&lang=de |
| P0 | hydro_reservoir,electricity_balance | karte-der-wasserkraftanlagen-der-schweiz |  | GEOTIFF | https://www.uvek-gis.admin.ch/BFE/ogd/120/ogd120_wasserkraftanlagen_karte.zip |
| P0 | hydro_reservoir,pv,prices_market | referenz-marktpreise-fur-die-gleitende-marktpramie-gemass-art-30aquinquies-enfv | Reference market price for the floating market premium | CSV,PDF,XML | https://www.bfe-ogd.ch/ogd125_gmp_quartalspreise.csv |
| P0 | installed_capacity | kernkraftwerke | Nuclear Power Plants | HTML,OCTET,XML | https://data.geo.admin.ch/ch.bfe.kernkraftwerke/kernkraftwerke/kernkraftwerke_2056.csv.zip |
| P0 | installed_capacity | konzept-windenergie-grundlagenkarte-des-bundes-betreffend-die-hauptsachlichen-windpotenzialgebi | Wind energy concept – Swiss federal government basic map of main areas with wind-power potential | HTML,OCTET,XML | https://wms.geo.admin.ch/?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities&lang=de |
| P0 | installed_capacity | sachplan-geologische-tiefenlager | Sectoral Plan Deep Geological Repositories | HTML,OCTET,XML | https://data.geo.admin.ch/ch.bfe.sachplan-geologie-tiefenlager/sachplan-geologie-tiefenlager/sachplan-geologie-tiefenlager_2056.gpkg |
| P0 | installed_capacity | windenergieanlagen | Wind energy plants | HTML,OCTET,XML | https://wms.geo.admin.ch/?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities&lang=de |
| P0 | installed_capacity,electricity_balance,pv,energiedashboard | energiedashboard-ch-stromproduktion-swissgrid | energiedashboard.ch: Electricity production Swissgrid | CSV | https://www.bfe-ogd.ch/ogd104_stromproduktion_swissgrid.csv |
| P0 | installed_capacity,pv | elektrizitatsproduktionsanlagen | Electricity production plants | HTML,OCTET,XML | https://data.geo.admin.ch/ch.bfe.elektrizitaetsproduktionsanlagen/csv/2056/ch.bfe.elektrizitaetsproduktionsanlagen.zip |
| P0 | installed_capacity,pv | photovoltaik-grossanlagen-in-der-schweiz | Large-scale photovoltaic systems in Switzerland | HTML,OCTET,XML | https://data.geo.admin.ch/ch.bfe.photovoltaik-grossanlagen/csv/2056/ch.bfe.photovoltaik-grossanlagen.zip |

## Production Use

This is a discovery register. A dataset becomes model-usable only after a dedicated importer, local raw cache, schema validation, vintage metadata and production gate documentation.
