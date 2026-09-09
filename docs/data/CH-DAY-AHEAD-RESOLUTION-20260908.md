# Swiss day-ahead resolution — verified 8 September 2026

Conclusion: current CH day-ahead reference is hourly. A2027 delay is supported
for cross-border capacity auctions; **end2027 is not confirmed**, and that
announcement is not a firm launch notice for the domestic EPEX energy auction.
Keep these distinctions explicit; do not introduce an automatic calendar switch.

Primary sources checked:

1. [EPEX SPOT index specification, July2026, p40](https://www.epexspot.com/sites/default/files/download_center_files/EPEX%20SPOT%20Indices%202019-05_final.pdf):
   CH day-ahead indices are defined on60min instruments. Page40 was visually
   checked as well as text-extracted. The URL retains an older filename while
   the document heading is July2026; do not infer its version from its URL.
2. [JAO,30June2026](https://www.jao.eu/news/update-st-auctions-swiss-borders-15-min-mtu-and):
   Swiss-border short-term15min MTU and block-bid target moves to2027 following
   testing and dependencies on LTFBA; no month or day is specified.
3. [JAO,19May2026](https://www.jao.eu/news/st-auctions-swiss-borders-15-min-mtu):
   preceding provisional schedule was delivery2September2026 for CH-FR/DE/IT,
   subject to tests/approval. The June update supersedes that schedule.
4. [JAO,7April2025](https://www.jao.eu/news/tsos-survey-introduction-15-minutes-mtu):
   distinguishes cross-border capacity auctions and parallel discussions with
   EPEX for domestic Swiss day-ahead energy trading. They are related projects,
   not interchangeable contractual dates.
5. [Swissgrid Balancing Roadmap2026–2030, pp24–25](https://www.swissgrid.ch/content/dam/swissgrid/about-us/newsroom/publications/balancing-roadmap-en.pdf):
   earlier Q3 2026 target must be read alongside the subsequent JAO delay.
6. [APG,2July2026](https://markt.apg.at/news-presse/update-of-st-auctions-on-swiss-borders-in-15-min-mtu-and-block-bids-go-live-timeline/):
   links to the JAO update, corroborating its primary-source identity.

EPEX market-result search pages sometimes returned a different area/modality
from their URL parameters; these ambiguous pages were not used as proof of CH
day-ahead quarter-hour resolution. No market-price scraping, new data ingestion,
Warehouse call or model input came from this web research.

Engineering consequence: score retained CH observations hourly, keep optional
15min shape experiments separate, and validate actual native interval metadata
when a future governed extract is acquired. Existing Swiss continuous intraday
15min trading does not turn the retained day-ahead observations into15min truth.
D307 keeps the existing point curves and monthly solver authorities unchanged.
