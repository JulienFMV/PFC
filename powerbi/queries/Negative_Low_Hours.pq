let
    RepoRoot = "H:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\",
    Source = Csv.Document(
        File.Contents(RepoRoot & "powerbi\data\negative_low_hours.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(
        Promoted,
        {
            {"year", Int64.Type},
            {"month", Int64.Type},
            {"hour", Int64.Type},
            {"rows", Int64.Type},
            {"weighted_negative_hours", Int64.Type},
            {"fast_negative_hours", Int64.Type},
            {"p10_negative_hours", Int64.Type},
            {"min_weighted_eur_mwh", type number},
            {"min_fast_eur_mwh", type number},
            {"min_p10_eur_mwh", type number}
        }
    )
in
    Typed
