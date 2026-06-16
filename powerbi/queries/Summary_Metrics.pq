let
    RepoRootNormalized = if Text.EndsWith(RepoRoot, "\") then RepoRoot else RepoRoot & "\",
    Source = Csv.Document(
        File.Contents(RepoRootNormalized & "powerbi\data\summary_metrics.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(
        Promoted,
        {
            {"metric", type text},
            {"value", type text}
        }
    )
in
    Typed
