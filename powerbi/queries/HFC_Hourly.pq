let
    RepoRootNormalized = if Text.EndsWith(RepoRoot, "\") then RepoRoot else RepoRoot & "\",
    Source = Csv.Document(
        File.Contents(RepoRootNormalized & "powerbi\data\hfc_hourly_powerbi.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(
        Promoted,
        {
            {"timestamp_ch_iso", type datetime},
            {"timestamp_ch", type text},
            {"utc_offset_ch", type text},
            {"timestamp_utc", type text},
            {"date_ch", type date},
            {"year", Int64.Type},
            {"quarter", Int64.Type},
            {"month", Int64.Type},
            {"month_label", type text},
            {"hour", Int64.Type},
            {"weekday", Int64.Type},
            {"season", type text},
            {"is_weekend", type logical},
            {"is_eex_peak", type logical},
            {"is_weighted_negative", type logical},
            {"is_fast_negative", type logical},
            {"is_p10_negative", type logical},
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
