import streamlit as st
import pandas as pd
import os
import sys
import importlib
import traceback

# Ensure Models directory is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Models'))

try:
    from data_splitter import split_and_save
except ImportError:
    st.error("Could not import split_and_save from data_splitter.py")

# Blinking button CSS
st.markdown("""
<style>
@keyframes pulse {
  0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
  70% { transform: scale(1.02); box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
  100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
}
.stButton>button[kind="primary"] {
    animation: pulse 2s infinite;
    background-color: #4CAF50 !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 10px 24px !important;
}
.stButton>button[kind="primary"]:hover {
    background-color: #45a049 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Auto-Model Evaluator Dashboard")

if 'step' not in st.session_state:
    st.session_state.step = 0

# --- STEP 1: Upload Data ---
st.header("Step 1: Upload & Clean Dataset")
uploaded_file = st.file_uploader("Drop a .csv, .xlsx, or .json file here", type=['csv', 'xlsx', 'json'])

if uploaded_file is not None:
    if 'df' not in st.session_state or st.session_state.get('curr_file') != uploaded_file.name:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                st.session_state.df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                st.session_state.df = pd.read_json(uploaded_file)
            st.session_state.curr_file = uploaded_file.name
            st.session_state.step = 0 # reset steps on new upload
        except Exception as e:
            st.error(f"Error reading file: {e}")
            
if 'df' in st.session_state:
    df = st.session_state.df
    st.write(f"**Data Preview ({df.shape[0]} rows, {df.shape[1]} columns):**")
    st.dataframe(df.head())
    
    st.subheader("Data Alterations")
    cols_to_drop = st.multiselect("Select columns to drop (optional)", df.columns)
    
    # Working copy
    df_clean = df.drop(columns=cols_to_drop) if cols_to_drop else df
    
    if cols_to_drop:
        st.write("Preview after dropping columns:")
        st.dataframe(df_clean.head())
    
    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("Select Target Column", df_clean.columns)
    with col2:
        task_type = st.radio("Select Task Type", ["regression", "classification"])

    st.subheader("📊 Feature Significance")
    try:
        from sklearn.feature_selection import f_classif, f_regression
        import numpy as np
        
        df_encoded = df_clean.copy()
        for c in df_encoded.columns:
            if df_encoded[c].dtype == 'object':
                df_encoded[c] = df_encoded[c].astype('category').cat.codes
                
        X_sig = df_encoded.drop(columns=[target_col]).fillna(0)
        y_sig = df_encoded[target_col].fillna(0)
        
        if task_type == 'classification':
            F, pval = f_classif(X_sig, y_sig)
        else:
            F, pval = f_regression(X_sig, y_sig)
            
        sig_df = pd.DataFrame({
            "Feature": X_sig.columns,
            "F-Value": F,
            "p-Value": pval
        }).sort_values('F-Value', ascending=False)
        
        st.dataframe(sig_df.style.background_gradient(subset=['F-Value'], cmap='viridis'))
    except Exception as e:
        st.warning(f"Could not compute feature significance: {e}")

    st.markdown("---")
    st.header("Step 2: Data Splitting Options")
    
    col3, col4 = st.columns(2)
    with col3:
        test_size = st.slider("Test Size Proportion", 0.05, 0.5, 0.2, 0.05)
    with col4:
        valid_size = st.slider("Validation Size Proportion", 0.05, 0.5, 0.2, 0.05)
        
    outputs_dir = os.path.join(os.path.dirname(__file__), 'temp_splits')
    os.makedirs(outputs_dir, exist_ok=True)
        
    if st.button("GO: Compute Splits & Select Models", type="primary" if st.session_state.step <= 1 else "secondary", key="split_go"):
        st.session_state.df_clean = df_clean
        st.session_state.target_col = target_col
        st.session_state.task_type = task_type
        with st.spinner("Splitting data..."):
            try:
                train_p, test_p, valid_p, train_l, test_l, valid_l = split_and_save(
                    df_clean, target_col, test_size, valid_size, outputs_dir, task_type
                )
                st.session_state.data_files = {
                    'train': train_p, 'test': test_p, 'valid': valid_p
                }
                st.session_state.split_info = {
                    'train_len': train_l, 'test_len': test_l, 'valid_len': valid_l
                }
                st.session_state.step = max(st.session_state.step, 2)
                st.rerun()
            except Exception as e:
                st.error(f"Error during splitting: {e}")
                
    if 'split_info' in st.session_state:
        st.success(f"Split complete! Train: {st.session_state.split_info['train_len']}, Valid: {st.session_state.split_info['valid_len']}, Test: {st.session_state.split_info['test_len']} rows.")

# --- STEP 3: Configure Models ---
if st.session_state.step >= 2:
    st.markdown("---")
    st.header("Step 3: Select and Configure Models")
    
    # Dynamically find models in the directory
    models_dir = os.path.join(os.path.dirname(__file__), 'Models')
    available_files = [f for f in os.listdir(models_dir) if f.endswith('.py')]
    
    # Filter by task_type
    if st.session_state.task_type == 'regression':
        valid_models = [m.replace('.py', '') for m in available_files if 'Regressor_Eval' in m]
    else:
        valid_models = [m.replace('.py', '') for m in available_files if 'Classifier_Eval' in m or 'SVC' in m or 'NB' in m]
        
    selected_models = st.multiselect(
        f"Available {st.session_state.task_type.capitalize()} Models", 
        valid_models, 
        default=valid_models[:3] if len(valid_models) >= 3 else valid_models
    )
    
    if st.button("GO: Run Evaluation Pipeline", type="primary" if st.session_state.step == 2 else "secondary", key="eval_go"):
        st.session_state.selected_models = selected_models
        st.session_state.step = max(st.session_state.step, 3)
        st.rerun()

# --- STEP 4: Execution & Results ---
if st.session_state.step >= 3:
    st.markdown("---")
    st.header("Step 4: Pipeline Execution & Report")
    
    results = []
    
    if not st.session_state.selected_models:
        st.warning("Please select at least one model to evaluate.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, model_name in enumerate(st.session_state.selected_models):
            status_text.text(f"Evaluating {model_name}...")
            try:
                # Dynamic import
                module = importlib.import_module(model_name)
                # Infer class name
                class_name = model_name.replace('_', '')
                if hasattr(module, class_name):
                    evaluator_class = getattr(module, class_name)
                    # Pass default hyperparams format
                    default_hyperparams = {}
                    if 'Linear' in class_name: default_hyperparams={'fit_intercept': [True, False]}
                    elif 'RandomForest' in class_name: default_hyperparams={'n_estimators': [10, 20]}
                    elif 'LogisticRegression' in class_name: default_hyperparams={'C': [1.0]}
                    
                    evaluator = evaluator_class(st.session_state.data_files, default_hyperparams)
                    best_model, params, score = evaluator.tune_and_evaluate()
                    
                    results.append({
                        "Model": model_name.replace('_Eval', ''),
                        "Best Score": score,
                        "Best Params": str(params)
                    })
                else:
                    st.warning(f"Could not find class {class_name} in module {model_name}")
            except Exception as e:
                st.error(f"Error evaluating {model_name}: {e}")
                # traceback.print_exc() # For debug
                
            progress_bar.progress((i + 1) / len(st.session_state.selected_models))
            
        status_text.text("Evaluation Complete!")
        
        if results:
            results_df = pd.DataFrame(results).sort_values(by="Best Score", ascending=False)
            st.subheader("🏆 Leaderboard")
            st.dataframe(results_df, use_container_width=True)
            
            st.bar_chart(results_df.set_index('Model')['Best Score'])
            
        if st.button("Reset Pipeline"):
            st.session_state.clear()
            st.rerun()
