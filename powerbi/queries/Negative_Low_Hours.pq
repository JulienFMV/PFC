let
    RepoRootNormalized = if Text.EndsWith(RepoRoot, "\") then RepoRoot else RepoRoot & "\",
    Source = Csv.Document(
        File.Contents(RepoRootNormalized & "powerbi\data\negative_low_hours.csv"),
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
