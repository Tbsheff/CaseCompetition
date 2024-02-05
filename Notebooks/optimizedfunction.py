import pandas as pd
from scipy.stats import linregress, pearsonr, kendalltau, spearmanr
from concurrent.futures import ThreadPoolExecutor


def optimized_bivariate(df, label, roundto=4):
    """Optimized bivariate analysis function with statistical analyses."""

    def process_feature(feature, original_df, label, roundto):
        # Work with copies to avoid altering the original DataFrame
        feature_data = pd.to_numeric(original_df[feature], errors="coerce").dropna()
        label_data = pd.to_numeric(original_df[label], errors="coerce").dropna()

        # Ensure we're only working with rows where both feature and label data are present
        common_index = feature_data.index.intersection(label_data.index)
        feature_data = feature_data.loc[common_index]
        label_data = label_data.loc[common_index]

        missing = round(
            (original_df.shape[0] - len(common_index)) / original_df.shape[0] * 100,
            roundto,
        )
        dtype = feature_data.dtype
        unique = feature_data.nunique()

        # Initialize result dictionary
        result = {
            "feature": feature,
            "missing %": missing,
            "type": str(dtype),
            "unique": unique,
            "p": None,
            "r": None,
            "τ": None,
            "ρ": None,
            "y = m(x) + b": None,
            "F": None,
            "X2": None,
        }

        # Perform statistical analyses
        if pd.api.types.is_numeric_dtype(dtype):
            # Note: Ensure that `feature_data` and `label_data` are Series objects
            result["r"], result["p"] = pearsonr(feature_data, label_data)
            result["τ"], _ = kendalltau(feature_data, label_data)
            result["ρ"], _ = spearmanr(feature_data, label_data)
            slope, intercept, r_value, p_value, std_err = linregress(
                feature_data, label_data
            )
            result["y = m(x) + b"] = (
                f"y = {round(slope, roundto)}(x) + {round(intercept, roundto)}"
            )
            # Additional statistical tests would be added here

        return result

    output_columns = [
        "feature",
        "missing %",
        "type",
        "unique",
        "p",
        "r",
        "τ",
        "ρ",
        "y = m(x) + b",
        "F",
        "X2",
    ]

    # List of features to process, excluding the label
    features = [feature for feature in df.columns if feature != label]

    # Initialize an empty list to store result dictionaries
    results_list = []

    # Processing features in parallel and collecting results
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_feature, feature, df, label, roundto)
            for feature in features
        ]
        for future in futures:
            results_list.append(future.result())

    # Convert the list of dictionaries to a DataFrame
    output_df = pd.DataFrame(results_list)

    return output_df
