"""
Regression Analysis Tool (Streamlit Web App)

Instructions:
-------------
1. Install dependencies (ideally in a virtual environment):

    pip install pandas numpy scikit-learn statsmodels streamlit

2. Save this file as `app.py` (or any name you wish).

3. Run the Streamlit app:

    streamlit run app.py

Features:
---------
- Upload your CSV data
- Select dependent (Y) and independent (X) variables
- Apply optional log, square, and interaction transformations to X variables
- Adjust train-test split ratio
- View regression coefficients, in-sample and out-of-sample R², and predictions table
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


def reset_app_state_except_data():
    """Clear Streamlit session state except for the uploaded data.

    This is intended to be used as a button callback. The button click
    itself triggers a rerun, so we do not explicitly call st.rerun().
    """
    keep_keys = {"uploaded_df"}
    for key in list(st.session_state.keys()):
        if key not in keep_keys:
            del st.session_state[key]

# ----------------- Utility Functions -----------------

def safe_log(series, base='e'):
    """Return log or log10 of a pandas Series, or None if not valid (i.e., contains nonpositive values)."""
    if (series <= 0).any():
        return None
    if base == 'e':
        return np.log(series)
    elif base == '10':
        return np.log10(series)
    else:
        raise ValueError("Invalid base for log. Use 'e' or '10'.")

def get_transformed_columns(X_df, transformations):
    """
    Given a dataframe X_df and a dict:
        transformations: { X_col_name: [list of transformations] }
    returns (transformed_df, added_colnames, warnings)
    """
    transformed = X_df.copy()
    added_colnames = []
    warnings = []
    # transformations: dict with key: column name, value: list of transforms
    for col, transforms in transformations.items():
        source = X_df[col]
        for t in transforms:
            if t == 'ln':
                log_col_name = f"ln_{col}"
                res = safe_log(source, 'e')
                if res is None:
                    warnings.append(
                        f"Cannot apply natural log to '{col}' (contains zeros or negative values)."
                    )
                else:
                    transformed[log_col_name] = res
                    added_colnames.append(log_col_name)
            elif t == 'log10':
                log_col_name = f"log10_{col}"
                res = safe_log(source, '10')
                if res is None:
                    warnings.append(
                        f"Cannot apply log10 to '{col}' (contains zeros or negative values)."
                    )
                else:
                    transformed[log_col_name] = res
                    added_colnames.append(log_col_name)
            elif t == 'square':
                sq_col_name = f"{col}_squared"
                transformed[sq_col_name] = source ** 2
                added_colnames.append(sq_col_name)
    return transformed, added_colnames, warnings

def add_interaction_terms(df, interaction_pairs):
    """
    For each tuple (col1, col2) in interaction_pairs, add new column '{col1}_{col2}_interaction'
    Returns:
        df_with_interactions, added_interaction_colnames
    """
    df_int = df.copy()
    interaction_colnames = []
    for col1, col2 in interaction_pairs:
        inter_col = f"{col1}_{col2}_interaction"
        df_int[inter_col] = df_int[col1] * df_int[col2]
        interaction_colnames.append(inter_col)
    return df_int, interaction_colnames

# ----------------- Streamlit App -----------------

st.title("Regression Analysis Tool")
st.markdown("Upload your data, select variables and transformations, and perform OLS regression with train-test evaluation.")

# --------- 1. File Upload ---------

uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state["uploaded_df"] = df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()
elif "uploaded_df" in st.session_state:
    df = st.session_state["uploaded_df"]

if df is not None:
    # Show reset button and data preview
    st.button("Reset selections", on_click=reset_app_state_except_data)

    st.write("### Data Preview")
    st.dataframe(df.head())

    # Drop columns with all nulls, and alert if any
    orig_cols = df.columns.tolist()
    empty_cols = [col for col in orig_cols if df[col].dropna().empty]
    if empty_cols:
        st.warning(f"Columns with no data will be ignored: {', '.join(empty_cols)}")
        df = df.drop(columns=empty_cols)
    cols = df.columns.tolist()
    
    if len(cols) < 2:
        st.warning("The dataset must have at least two columns.")
        st.stop()

    st.write("#### Columns detected:")
    st.write(cols)

    # Check if all columns are numeric except possibly for Y
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.warning("There are not enough numeric columns in your data.")
        st.stop()

    # --------- 2. Variable Selection ---------

    st.write("## 1. Select Variables")
    y_col = st.selectbox("Select the dependent (target) variable (Y):", options=cols)

    # By default, allow only numeric columns for X (excluding Y)
    possible_x = [col for col in num_cols if col != y_col]
    if not possible_x:
        st.error("No suitable independent (X) variables found (numeric, except target Y).")
        st.stop()
    x_cols = st.multiselect("Select independent variable(s) (X):", options=possible_x, default=possible_x)
    if not x_cols:
        st.warning("Select at least one independent variable.")
        st.stop()

    # --------- 3. Y Transformation (optional) ---------

    st.write("## 2. Y Transformation (optional)")
    y_transform_options = ['None', 'ln', 'log10', 'square']
    y_transformation_labels = {
        'None': 'No transformation',
        'ln': 'Natural log of Y',
        'log10': 'Base-10 log of Y',
        'square': 'Square of Y',
    }
    y_transform_choice = st.selectbox(
        "Select transformation for Y (optional):",
        options=y_transform_options,
        format_func=lambda x: y_transformation_labels[x]
    )

    # --------- 4. X Transformations ---------

    st.write("## 3. X Variable Transformations (optional)")

    transform_options = ['ln', 'log10', 'square']

    x_transforms = {}
    drop_original_after_transform = {}
    transformation_labels = {'ln': 'Natural log', 'log10': 'Base-10 log', 'square': 'Square'}

    for col in x_cols:
        with st.expander(f"Transformations for {col}"):
            selected_t = st.multiselect(
                f"Select transformations for '{col}'", options=transform_options, format_func=lambda x: transformation_labels[x]
            )
            x_transforms[col] = selected_t

            if selected_t:
                drop_flag = st.checkbox(
                    f"Drop original '{col}' from model after applying transformations",
                    value=False,
                    key=f"drop_orig_{col}"
                )
            else:
                drop_flag = False
            drop_original_after_transform[col] = drop_flag

    # --------- 5. Interaction Effects ---------

    st.write("## 4. Interaction Terms (optional)")

    # Present all pairwise combinations of the selected X's (unordered pairs: (i < j))
    if len(x_cols) > 1:
        import itertools
        pairs = list(itertools.combinations(x_cols, 2))
        pair_labels = [f"{a} × {b}" for (a, b) in pairs]
        selected_pairs = st.multiselect(
            "Select interactions to include (multiplicative):",
            options=pairs,
            format_func=lambda tup: f"{tup[0]} × {tup[1]}"
        )
    else:
        selected_pairs = []

    # --------- 6. Train-Test Split ---------

    st.write("## 5. Train-Test Split Settings")
    split_ratio = st.slider("Test set fraction", min_value=0.1, max_value=0.9, value=0.3, step=0.05)
    split_train = 1 - split_ratio

    # --------- 7. Feature Engineering ---------

    # Start from selected columns
    df_model = df[[y_col] + x_cols].copy()

    # Apply Y transformation if requested
    y_raw = df_model[y_col]
    y_transform_applied = "None"
    if y_transform_choice == 'ln':
        y_trans = safe_log(y_raw, 'e')
        if y_trans is None:
            st.error(f"Cannot apply natural log to Y ('{y_col}') because it contains zeros or negative values.")
            st.stop()
        df_model[y_col] = y_trans
        y_transform_applied = "ln"
    elif y_transform_choice == 'log10':
        y_trans = safe_log(y_raw, '10')
        if y_trans is None:
            st.error(f"Cannot apply log10 to Y ('{y_col}') because it contains zeros or negative values.")
            st.stop()
        df_model[y_col] = y_trans
        y_transform_applied = "log10"
    elif y_transform_choice == 'square':
        df_model[y_col] = y_raw ** 2
        y_transform_applied = "square"

    # Drop any remaining missing values after transformation
    df_model = df_model.dropna()

    original_X = x_cols.copy()
    # Apply X transformations
    X0 = df_model[x_cols]
    X_wt, transformed_variables, transform_warnings = get_transformed_columns(X0, x_transforms)
    all_X_vars = x_cols + transformed_variables

    # Optionally drop original variables that have been transformed
    dropped_originals = []
    for col in x_cols:
        if drop_original_after_transform.get(col) and x_transforms.get(col):
            if col in X_wt.columns:
                X_wt = X_wt.drop(columns=[col])
            if col in all_X_vars:
                all_X_vars.remove(col)
            dropped_originals.append(col)

    # Apply interaction terms (on transformed X columns + added features)
    X_wt, interaction_variables = add_interaction_terms(X_wt, selected_pairs)
    all_X_vars += interaction_variables

    # Warnings
    if transform_warnings:
        for w in transform_warnings:
            st.warning(w)
    if not all_X_vars:
        st.error("No valid independent variables to use after transformations/interactions.")
        st.stop()

    feats_summary = []
    feats_summary.append(f"Y variable: {y_col} (transformation: {y_transform_applied})")
    if original_X:
        feats_summary.append(f"Original X vars: {', '.join(original_X)}")
    if dropped_originals:
        feats_summary.append(f"Original variables dropped after transformation: {', '.join(dropped_originals)}")
    if transformed_variables:
        feats_summary.append(f"Transformed variables: {', '.join(transformed_variables)}")
    if interaction_variables:
        feats_summary.append(f"Interaction terms: {', '.join(interaction_variables)}")
    st.info('\n'.join(feats_summary))

    # --------- 7. Model Execution Button ---------

    if st.button("Run Regression"):
        #----------- Split Data --------------

        # Get clean data (all input features, target)
        X_all = X_wt[all_X_vars]
        y_all = df_model[y_col].loc[X_wt.index]  # Align index just in case

        # Make sure all columns are numeric!
        if not np.all([np.issubdtype(X_all[col].dtype, np.number) for col in X_all.columns]):
            st.error("All features must be numeric after transformation. Please check your variables.")
            st.stop()

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=split_ratio, random_state=42
        )

        n_train, n_test = X_train.shape[0], X_test.shape[0]
        if n_train < 2 or n_test < 1:
            st.error(f"Too few observations in training (got {n_train}) or test (got {n_test}) set for regression to run.")
            st.stop()

        #----------- Model Fitting --------------

        # Add constant for intercept
        X_train_const = sm.add_constant(X_train, has_constant='add')
        model = sm.OLS(y_train, X_train_const)
        results = model.fit()

        # In-sample R²
        r2_train = results.rsquared

        # Predict on test
        X_test_const = sm.add_constant(X_test, has_constant='add')
        # Ensure columns align
        X_test_const = X_test_const[X_train_const.columns]
        y_pred_test = results.predict(X_test_const)

        # Out-of-sample R² as specified
        ss_res = ((y_test - y_pred_test) ** 2).sum()
        ss_tot = ((y_test - y_test.mean()) ** 2).sum()
        r2_test = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        # ---------- Output Section ---------------

        st.write("## Regression Results")
        st.write(f"**Y variable:** {y_col}  |  **Y transformation:** {y_transform_applied}")

        # Coefficient table
        coef_df = pd.DataFrame({
            'Variable': X_train_const.columns,
            'Coefficient': results.params.values,
            'Std. Error': results.bse.values,
            'P-value': np.round(results.pvalues.values, 3)
        })
        st.write("### Coefficients (including intercept):")
        st.dataframe(coef_df, width='stretch')

        # In-sample & out-of-sample metrics
        st.write("### Model Evaluation")
        st.write(f"**In-sample R² (train)**: {r2_train:.4f}")
        st.write(f"**Out-of-sample R² (test)**: {r2_test:.4f}")
        st.write(f"**Training observations:** {n_train}")
        st.write(f"**Test observations:** {n_test}")

        # Feature breakdown
        st.write("### Feature Summary:")
        st.write("**Original X variables:**", ', '.join(original_X))
        st.write("**Transformed variables added:**", ', '.join(transformed_variables) if transformed_variables else "None")
        st.write("**Interaction terms added:**", ', '.join(interaction_variables) if interaction_variables else "None")

        # Test set predictions (table)
        pred_df = pd.DataFrame({
            'Actual (y)': y_test,
            'Predicted (y_hat)': y_pred_test
        })
        st.write("### Actual vs Predicted (Test Set):")
        st.dataframe(pred_df.reset_index(drop=True), width='stretch')

        # Also show fitted values for train in a separate expandable table
        with st.expander("Show Training Set: Actual vs Fitted", expanded=False):
            y_fitted_train = results.fittedvalues
            train_pred_df = pd.DataFrame({
                'Actual (y)': y_train,
                'Fitted (y_hat)': y_fitted_train
            })
            st.dataframe(train_pred_df.reset_index(drop=True), width='stretch')

        # Optional: download results
        import io
        with st.expander("Download Coefficients Table"):
            csv_file = coef_df.to_csv(index=False)
            st.download_button(
                label="Download coefficients as CSV",
                data=csv_file,
                file_name="regression_coefficients.csv",
                mime="text/csv",
            )

else:
    st.info("Please upload a CSV file to begin.")

# End Generation Here