let
    RepoRootNormalized = if Text.EndsWith(RepoRoot, "\") then RepoRoot else RepoRoot & "\",
    Source = Csv.Document(
        File.Contents(RepoRootNormalized & "powerbi\data\eex_residuals.csv"),
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
