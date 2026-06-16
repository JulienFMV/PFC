let
    RepoRoot = "H:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\",
    Source = Csv.Document(
        File.Contents(RepoRoot & "powerbi\data\summary_metrics.csv"),
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
