let
    RepoRoot = "H:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\",
    Source = Csv.Document(
        File.Contents(RepoRoot & "powerbi\data\duck_month_hour.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(
        Promoted,
        {
            {"year", Int64.Type},
            {"month", Int64.Type},
            {"hour", Int64.Type},
            {"price_slow_eur_mwh", type number},
            {"price_central_eur_mwh", type number},
            {"price_fast_eur_mwh", type number},
            {"price_weighted_mean_eur_mwh", type number},
            {"structural_p10_eur_mwh", type number},
            {"structural_p50_eur_mwh", type number},
            {"structural_p90_eur_mwh", type number},
            {"structural_width_eur_mwh", type number}
        }
    )
in
    Typed
