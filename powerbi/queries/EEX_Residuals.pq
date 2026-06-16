let
    RepoRoot = "H:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\",
    Source = Csv.Document(
        File.Contents(RepoRoot & "powerbi\data\eex_residuals.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(
        Promoted,
        {
            {"load_type", type text},
            {"product", type text},
            {"target_eex_eur_mwh", type number},
            {"csv_mean_eur_mwh", type number},
            {"abs_error_eur_mwh", type number},
            {"rows", Int64.Type}
        }
    )
in
    Typed
