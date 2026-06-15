# Phase 13 Literature: Structural Electrification Shapes

This file lists the scientific and institutional references that should anchor
the Phase 13 implementation. Use them for model design, not as decoration.

## Core Price-Formation Literature

### Sensfuss, Ragwitz, Genoese (2008)

Reference: Frank Sensfuss, Mario Ragwitz, Massimo Genoese, "The merit-order
effect: A detailed analysis of the price effect of renewable electricity
generation on spot market prices in Germany", Energy Policy 36, 3086-3094.

Use in Phase 13: renewable generation shifts the supply stack and depresses
prices in the hours where it is abundant. This supports a PV-driven midday
price-shape term.

URL: https://publikationen.bibliothek.kit.edu/1000016665

### Hirth (2013)

Reference: Lion Hirth, "The market value of variable renewables: The effect of
solar wind power variability on their relative price", Energy Economics 38,
218-236.

Use in Phase 13: value factors decline with penetration. This is the economic
foundation for PV cannibalization and the need to express PV penetration as a
structural shape driver.

URL: https://publications.pik-potsdam.de/pubman/faces/ViewItemFullPage.jsp?itemId=item_18936_1

### Gowrisankaran, Reynolds, Samano (2016)

Reference: Gautam Gowrisankaran, Stanley S. Reynolds, Mario Samano,
"Intermittency and the Value of Renewable Energy", Journal of Political Economy
124(4), 1187-1234.

Use in Phase 13: intermittency has economic value and cost effects beyond
average generation. This supports separating structural shape risk from normal
spot residual volatility.

URL: https://www.journals.uchicago.edu/doi/full/10.1086/686733

## Duck Curve and Storage

### Denholm, Brinkman, Jorgenson (2015)

Reference: Paul Denholm, Gregory Brinkman, Jennie Jorgenson, "Overgeneration
from Solar Energy in California: A Field Guide to the Duck Chart", NREL.

Use in Phase 13: canonical operational treatment of solar overgeneration,
midday net-load depression, evening ramps, curtailment, and flexibility.

URL: https://research-hub.nlr.gov/en/publications/overgeneration-from-solar-energy-in-california-a-field-guide-to-t/

### Seel, Mills, Wiser (2018)

Reference: Joachim Seel, Andrew D. Mills, Ryan Wiser, "Impacts of High Variable
Renewable Energy Futures on Wholesale Electricity Prices, and on Electric-Sector
Decision Making", LBNL.

Use in Phase 13: high-VRE futures alter hourly wholesale price patterns. This is
directly relevant to long-term HPFC hourly-shape migration.

URL: https://www.osti.gov/biblio/1437006

### Schmalensee (2022)

Reference: Richard Schmalensee, "Competitive Energy Storage and the Duck Curve",
The Energy Journal 43(2), 1-16.

Use in Phase 13: storage is an economic response to the duck-curve problem.
Battery terms should compress spreads and refill the belly only when power and
energy capacity are sufficient.

URL: https://ideas.repec.org/a/sae/enejou/v43y2022i2p1-16.html

## Forecasting and HPFC Construction

### Kiesel, Paraschiv, Saethero

Reference: Ruediger Kiesel, Florentina Paraschiv, Audun Saethero, "On the
Construction of Hourly Price Forward Curves for Electricity Prices".

Use in Phase 13: HPFC construction is a constrained shape problem. Structural
shape terms must preserve quoted forward products.

URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2845302

### Benth, Koekebakker (2008)

Reference: Fred Espen Benth, Steen Koekebakker, "Stochastic modeling of
financial electricity contracts", Energy Economics 30(3), 1116-1157.

Use in Phase 13: electricity forwards have delivery periods; the implementation
must respect period-delivery products rather than treating the HPFC as a generic
spot forecast.

URL: https://ideas.repec.org/a/eee/eneeco/v30y2008i3p1116-1157.html

### Lago, Marcjasz, De Schutter, Weron (2021)

Reference: Jesus Lago, Grzegorz Marcjasz, Bart De Schutter, Rafal Weron,
"Forecasting day-ahead electricity prices: A review of state-of-the-art
algorithms, best practices and an open-access benchmark", Applied Energy 293.

Use in Phase 13: benchmark discipline. New complexity must beat strong
baselines, use robust evaluation, and avoid one-off private-data claims.

URL: https://arxiv.org/abs/2008.08004

### Marcjasz, Narajewski, Weron, Ziel (2023)

Reference: Grzegorz Marcjasz, Michal Narajewski, Rafal Weron, Florian Ziel,
"Distributional neural networks for electricity price forecasting", Energy
Economics 125.

Use in Phase 13: distributional EPF with modern methods. Useful for uncertainty
thinking, but do not cite it as proof of long-term seasonal regime contamination
unless a more specific source is found.

URL: https://dsee.wiwi.uni-due.de/en/research/publications/distributional-neural-networks-for-electricity-price-forecasting-16636/

## Scenario and Data Sources

### ENTSO-E / ENTSOG TYNDP 2024

Reference: TYNDP 2024 scenarios, including National Trends+, Distributed Energy,
and Global Ambition.

Use in Phase 13: official European scenario backbone for 2030, 2040, 2050,
including electricity, hydrogen, gas, demand, RES, and flexibility assumptions.

URL: https://2024.entsos-tyndp-scenarios.eu/tyndp-2024-scenarios/

### OFEN Energieperspektiven 2050+

Reference: Swiss federal long-term energy scenarios.

Use in Phase 13: Swiss official trajectory source for electrification, demand,
PV, heating, and system transformation. Use publication dates and scenario names
as vintage metadata.

URL: https://www.bfe.admin.ch/bfe/en/home/policy/energy-perspectives-2050-plus.html

### Pronovo

Reference: Swiss renewable installation and production statistics.

Use in Phase 13: realized CH renewable deployment and monthly production
profiles. Pronovo data is suitable for as-of actualization, subject to lags and
publication timestamps.

URL: https://pronovo.ch/

### Bundesnetzagentur Marktstammdatenregister

Reference: German market master data register for generation and storage assets.

Use in Phase 13: DE PV and battery actualization. This is important for CH
cross-border and basis effects.

URL: https://www.bundesnetzagentur.de/DE/Fachthemen/ElektrizitaetundGas/Monitoringberichte/Marktstammdatenregister/start.html

## Literature Hygiene

Avoid unsupported claims:

* "canyon curve" is useful industry language, but not a stable academic model.
  Use it as descriptive shorthand only.
* Do not claim battery deployment always refills the belly. It does so only if
  storage is installed, connected, dispatchable, and exposed to prices.
* Do not cite Marcjasz/Narajewski/Weron/Ziel 2025 for regime contamination
  unless a precise, verifiable DOI or preprint is available.

