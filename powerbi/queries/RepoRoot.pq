let
    // Power BI parameter/query used by the standalone query templates.
    // The build scripts regenerate the semantic model with the local absolute
    // path automatically, so normal users should run build_and_open_pfc_qa.ps1.
    // If using these .pq files manually, set this value to the repo root.
    Source = "<SET_REPO_ROOT_HERE>\"
in
    Source
