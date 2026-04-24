# Präsentation für Deviwa — einfacher deutscher Text

Pro Bildschirm: 3–5 Sätze zum Vorlesen, dann Stichworte zum Erklären.
Alles in einfachem Deutsch. Langsam sprechen. Bei Fragen auf Französisch
oder Englisch ausweichen.

---

## 🎯 1. Portfolio-Cockpit (Startseite)

**Zum Vorlesen:**

> «Guten Tag. Das ist Ihr Portfolio in einem Blick.
> Sie sehen Ihr Volumen, Ihren Wert und Ihr Risiko.
> Grün ist gut. Rot ist Achtung.
> Wir können auf einen Akteur filtern oder den ganzen Pool zeigen.»

**Was zeigen (in Reihenfolge, Mauszeiger folgen):**

1. Oben: **Brutto-Volumen** — das ist Ihr Gesamtvolumen in MWh.
2. **Netto-Position** — Einspeisung minus Bezug.
3. **Notional** — Wert aller Deals zum Einkaufs-Preis.
4. **Mark-to-Market** — Wert zum **heutigen Marktpreis**.
5. **Hedge-Quote** — wie viel Sie bereits gesichert haben.
   Ziel ist **70 Prozent**. Grün bedeutet: gut gesichert.
6. **Ø Hedge-Preis** vs **Markt-Preis** — Haben Sie gut eingekauft?
7. **Realisierter P&L** — fertig gelieferte Geschäfte.
   **Unrealisierter P&L** — noch offene Geschäfte.

**Wichtiger Satz:**
> «Alle diese Zahlen werden **täglich automatisch** aktualisiert,
> wenn Sie unser System nutzen.»

---

## 📊 2. Marktüberblick

**Zum Vorlesen:**

> «Hier sehen Sie den Markt von heute.
> Der Spot-Preis, die Prognose für morgen, und die Forwards.
> Alles auf einer Seite.»

**Was zeigen:**

1. Kachel **CH Spot** — letzter Stundenpreis, Vergleich zu gestern.
2. Kachel **Prognose D+1** — unser Modell für morgen.
3. Kachel **EEX Cal-27** — der Jahresforward 2027.
4. **Gas** und **CO₂** — wichtige Treiber für den Strompreis.
5. Unten: **14 Tage Spot + 10 Tage Prognose** mit blauem Band.
   Das Band zeigt die **Unsicherheit**.

**Wichtiger Satz:**
> «Das blaue Band ist unsere Unsicherheit.
> **Schmal** bedeutet sicher. **Breit** bedeutet risikoreich.»

---

## 📈 3. Kurzfristprognose

**Zum Vorlesen:**

> «Unser Prognose-Modell heisst **LEAR**.
> Es lernt jeden Tag neu.
> Wir prüfen die Qualität jede Woche.»

**Was zeigen:**

1. **MAE letzte 30 Tage** — das ist der Fehler in EUR pro MWh.
   Je **kleiner**, desto besser.
2. **10-Tage-Prognose** mit Band.
3. **Tabelle der Stunden morgen** — für Ihre Planung.
4. Unten: **realisierter Preis vs. Prognose**.
   Zeigt, wie nah unser Modell liegt.

**Wichtiger Satz:**
> «Wir verstecken nichts. Sie sehen die **echte Qualität** jeden Tag.»

---

## 💡 4. Lastprofil-Pricing

**Zum Vorlesen:**

> «Sie geben mir Ihre Lastkurve. Ich sage Ihnen den Preis.
> In einer Minute. Direkt hier im Browser.»

**Was zeigen (interaktiv!):**

1. **Datei hochladen** — CSV oder Excel. Oder **Beispielprofil** benutzen.
2. **Jahresvolumen** erscheint sofort.
3. **Profilpreis** — Ihr Preis pro MWh.
4. **Profilaufschlag** — was Ihr Profil teurer macht als Baseload.
5. **Gesamtkosten in EUR** — Ihre Strom-Rechnung für das Jahr.
6. **Peak-Anteil** — wie viel tagsüber.
7. Unten: **Monatsverlauf** Volumen und Preis.

**Wichtiger Satz:**
> «So wissen Sie **heute**, was Sie **nächstes Jahr** bezahlen werden.
> Keine Überraschung.»

---

## 📒 5. Ihre Transaktionen

**Zum Vorlesen:**

> «Hier sind **Ihre Geschäfte**.
> Ein Akteur oder der ganze Pool.
> Filter, Export, alles möglich.»

**Was zeigen:**

1. **RELL** wählen — Sie sehen nur RELL.
2. Dann **EDSH** — Sie sehen nur EDSH.
3. Dann **Gesamter Pool**.
4. Oben: **Anzahl Deals**, **Volumen**, **P&L**.
5. Mitte: Volumen und P&L **pro Monat**.
6. Unten: **alle Deals** — sortierbar, filterbar.
7. **CSV-Export**-Knopf unten rechts.

**Wichtiger Satz:**
> «Jeder Akteur sieht **nur seine Daten**, wenn er eingeloggt ist.
> Im Pool-Modus sehen wir alles zusammen.»

---

## 🎚️ 6. Programmqualität

**Zum Vorlesen:**

> «Haben Sie gut geplant?
> Wir vergleichen **Ihr Programm** mit der **Realität**.
> Abweichungen kosten Geld — Ausgleichsenergie.»

**Was zeigen:**

1. **Genauigkeit** — wie nah waren Sie am Plan? Grün ab 90 %.
2. **MAE** — durchschnittliche Abweichung in MW.
3. **Bias** — waren Sie immer zu hoch oder zu tief?
4. **Geschätzte Ausgleichskosten** — das Risiko in Euro.
5. Mitte: **Zeitreihe Plan gegen Ist**.
   Rote Linie = Abweichung.
6. **Streudiagramm** — jeder Punkt ist eine Stunde.
   Nah an der roten Linie = perfekt.
7. **Monatliche Genauigkeit** — Ziel 90 %.

**Wichtiger Satz:**
> «Wir zeigen Ihnen, **wo** und **wann** Sie verbessern können.
> Das spart direkt Geld.»

---

## 🛡️ 7. VaR und Stresstest

**Zum Vorlesen:**

> «Wie viel Geld kann ich verlieren, wenn der Markt schlecht läuft?
> Das ist Risikomanagement.
> Banken und Energieversorger nutzen dieselbe Methode.»

**Was zeigen:**

1. **Konfidenzniveau** wählen: 95 %, 97.5 % oder 99 %.
2. **Horizont**: 1 Tag, 5 Tage, 10 Tage.
3. **VaR** — Ihre maximale Verlustmöglichkeit.
4. **Expected Shortfall** — der Durchschnitt der schlechtesten Fälle.
5. **Tabelle Stresstests**:
   - Preis + 20 Prozent
   - Extremwinter + 50 Prozent
   - Trockenes Jahr + 15 Prozent
   - Nuklear-Ausfall + 30 Prozent
6. Unten: **historische Verteilung** aller möglichen Ergebnisse.

**Wichtiger Satz:**
> «Sie wissen **vor** dem Ereignis, was passieren kann.
> Das ist moderne Risikoführung.»

---

## 🌍 8. Marktdaten (Bonus)

**Zum Vorlesen:**

> «Zum Schluss: die Welt um uns herum.
> Strom, Wind, Sonne, Speicher.
> Alles was die Preise bewegt.»

**Was zeigen:**

1. **Last CH** — Stromverbrauch der Schweiz.
2. **Solar** und **Wind** — erneuerbare Einspeisung.
3. **Grenzflüsse** — Import und Export zu Nachbarländern.
4. **Speicher-Füllstand** — besonders **Wallis-Speicher** = 55 % der Schweiz.

**Wichtiger Satz:**
> «Wir sind **in der Region**.
> Wir kennen die Wallis-Stauseen. Das ist unser Vorteil.»

---

## Abschluss-Sätze (nach der Demo)

> «Das war die Demo.
>
> **Was Sie gesehen haben, läuft schon bei FMV intern.**
>
> Wir bauen daraus ein Portal für Sie.
> **Jeder Akteur hat seinen eigenen Zugang.**
> Daten bleiben in der Schweiz.
>
> **Zeitplan:** in 6 bis 8 Wochen ein erstes Live-Portal.
>
> **Preis:** wir besprechen das gerne nach der Demo.
>
> Haben Sie Fragen?»

---

## Schnelle Antworten auf typische Fragen

| Frage (DE) | Antwort (DE, einfach) |
|---|---|
| Wie genau ist die Prognose? | MAE ist zirka X EUR/MWh über 30 Tage. Wir messen es jeden Tag. |
| Wer sieht meine Daten? | Nur Sie. Wir nicht. Die anderen Akteure nicht. Ausnahme: Pool-Ansicht. |
| Wo laufen die Daten? | In der Schweiz, auf FMV-Servern. |
| Was kostet das? | Wir machen ein Angebot pro Akteur. Nach Volumen. |
| Kann ich das in mein System einbauen? | Ja. Wir liefern eine API. |
| Ist das sicher? | Ja. SSL, Login, Logs. Zugang nur mit Konto. |
| Was, wenn FMV ausfällt? | Ihre Daten sind in Ihrem System gespiegelt. |

---

## Aussprache-Hilfen (für Sie!)

- **Lastprofil** = "last-pro-FIL"
- **Kurzfristprognose** = "KURZ-frist-pro-GNO-se"
- **Programmqualität** = "pro-GRAMM-kva-li-TET"
- **Stresstest** = "SHTRESS-test"
- **Hedge-Quote** = "HEDGE-kvo-te" (man sagt "hedge" englisch)
- **Ausgleichsenergie** = "AUS-gleichs-e-NER-gee"
- **Mark-to-Market** = einfach englisch aussprechen
- **Erneuerbare** = "er-NOI-er-BA-re"

**Tipp:** Wenn Sie ein Wort nicht kennen — **zeigen Sie auf den Bildschirm**.
Die Zahlen sprechen für sich.
