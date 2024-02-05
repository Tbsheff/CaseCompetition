import pandas as pd
from scipy import stats
from scipy.stats import (
    linregress,
    pearsonr,
    kendalltau,
    spearmanr,
    f_oneway,
    chi2_contingency,
)
from concurrent.futures import ThreadPoolExecutor


def optimized_bivariate(df, label, roundto=4):
    """Optimized bivariate analysis function with statistical analyses."""

    def process_feature(feature, df, label, roundto):
        """Function to process each feature against the label, including statistical analyses."""
        df_temp = df[[feature, label]].dropna()
        missing = round((df.shape[0] - df_temp.shape[0]) / df.shape[0] * 100, roundto)
        dtype = df_temp[feature].dtype
        unique = df_temp[feature].nunique()

        # Initialize result dictionary to store the calculations
        result = {
            "feature": feature,
            "missing %": missing,
            "type": dtype,
            "unique": unique,
            "p": None,
            "r": None,
            "τ": None,
            "ρ": None,
            "y = m(x) + b": None,
            "F": None,
            "X2": None,
        }

        if pd.api.types.is_numeric_dtype(dtype) and pd.api.types.is_numeric_dtype(
            df[label].dtype
        ):
            # Pearson's r, Kendall's τ, Spearman's ρ
            result["r"], result["p"] = pearsonr(df_temp[feature], df_temp[label])
            result["τ"], _ = kendalltau(df_temp[feature], df_temp[label])
            result["ρ"], _ = spearmanr(df_temp[feature], df_temp[label])

            # Linear regression (slope m, intercept b)
            slope, intercept, r_value, p_value, std_err = linregress(
                df_temp[feature], df_temp[label]
            )
            result["y = m(x) + b"] = (
                f"y = {round(slope, roundto)}(x) + {round(intercept, roundto)}"
            )

            # Assuming the need for F-statistic and Chi-squared statistic would depend on the specific analysis.
            # These statistics are typically used in specific contexts (ANOVA for F, categorical analysis for Chi-squared)
            # Placeholder for F-statistic (ANOVA) - adjust based on actual analysis needs
            # result["F"], _ = f_oneway(...)

            # Placeholder for Chi-squared - adjust based on actual analysis needs
            # result["X2"], _, _, _ = chi2_contingency(...)

        return result

    # Prepare output DataFrame
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
    output_df = pd.DataFrame(columns=output_columns)

    # List of features to process, excluding the label
    features = [feature for feature in df.columns if feature != label]

    # Processing features in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_feature, feature, df, label, roundto)
            for feature in features
        ]
        results = [future.result() for future in futures]

    # Inside the optimized_bivariate function, after computing results in parallel

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


# Adjust the placeholders for F-statistic and Chi-squared statistic based on your specific analysis needs.
