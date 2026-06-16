let
    RepoRoot = "H:\Energy\GeCom\CONTROLLING RISK\Analyses diverses\Python - JB\PFC_LT\",
    DataRoot = RepoRoot & "powerbi\data\",

    LoadCsv = (fileName as text) as table =>
        let
            Source = Csv.Document(
                File.Contents(DataRoot & fileName),
                [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]
            ),
            Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true])
        in
            Promoted,

    HFC_Hourly_Raw = LoadCsv("hfc_hourly_powerbi.csv"),
    HFC_Hourly = Table.TransformColumnTypes(
        HFC_Hourly_Raw,
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
    ),

    Period_Means = Table.TransformColumnTypes(
        LoadCsv("period_means.csv"),
        {
            {"year", Int64.Type},
            {"quarter", Int64.Type},
            {"month", Int64.Type},
            {"price_slow_eur_mwh", type number},
            {"price_central_eur_mwh", type number},
            {"price_fast_eur_mwh", type number},
            {"price_weighted_mean_eur_mwh", type number},
            {"structural_p10_eur_mwh", type number},
            {"structural_p50_eur_mwh", type number},
            {"structural_p90_eur_mwh", type number},
            {"structural_width_eur_mwh", type number}
        }
    ),

    Duck_Month_Hour = Table.TransformColumnTypes(
        LoadCsv("duck_month_hour.csv"),
        {
            {"year", Int64.Type},
            {"month", Int64.Type},
            {"hour", Int64.Type},
            {"price_weighted_mean_eur_mwh", type number},
            {"structural_width_eur_mwh", type number}
        }
    ),

    Duck_Season_Hour = Table.TransformColumnTypes(
        LoadCsv("duck_season_hour.csv"),
        {
            {"year", Int64.Type},
            {"season", type text},
            {"hour", Int64.Type},
            {"price_weighted_mean_eur_mwh", type number},
            {"structural_width_eur_mwh", type number}
        }
    ),

    Negative_Low_Hours = Table.TransformColumnTypes(
        LoadCsv("negative_low_hours.csv"),
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
    ),

    Annual_Shape = Table.TransformColumnTypes(
        LoadCsv("annual_shape.csv"),
        {
            {"year", Int64.Type},
            {"mean_eur_mwh", type number},
            {"evening_minus_midday_eur_mwh", type number},
            {"weekend_minus_weekday_eur_mwh", type number},
            {"structural_width_mean_eur_mwh", type number},
            {"structural_width_p95_eur_mwh", type number}
        }
    ),

    EEX_Residuals = Table.TransformColumnTypes(
        LoadCsv("eex_residuals.csv"),
        {
            {"load_type", type text},
            {"product", type text},
            {"target_eex_eur_mwh", type number},
            {"csv_mean_eur_mwh", type number},
            {"abs_error_eur_mwh", type number},
            {"rows", Int64.Type}
        }
    ),

    Summary_Metrics = Table.TransformColumnTypes(
        LoadCsv("summary_metrics.csv"),
        {
            {"metric", type text},
            {"value", type text}
        }
    )
in
    [
        HFC_Hourly = HFC_Hourly,
        Period_Means = Period_Means,
        Duck_Month_Hour = Duck_Month_Hour,
        Duck_Season_Hour = Duck_Season_Hour,
        Negative_Low_Hours = Negative_Low_Hours,
        Annual_Shape = Annual_Shape,
        EEX_Residuals = EEX_Residuals,
        Summary_Metrics = Summary_Metrics
    ]
