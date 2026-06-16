let
    RepoRoot = "H:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\",
    Source = Csv.Document(
        File.Contents(RepoRoot & "powerbi\data\annual_shape.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(
        Promoted,
        {
            {"year", Int64.Type},
            {"mean_eur_mwh", type number},
            {"evening_minus_midday_eur_mwh", type number},
            {"weekend_minus_weekday_eur_mwh", type number},
            {"structural_width_mean_eur_mwh", type number},
            {"structural_width_p95_eur_mwh", type number}
        }
    )
in
    Typed
