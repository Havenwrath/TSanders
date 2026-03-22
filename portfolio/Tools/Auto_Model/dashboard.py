import streamlit as st
import pandas as pd
import os
import sys
import importlib
import numpy as np

# Ensure Models directory is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Models'))

try:
    from data_splitter import split_and_save
except ImportError:
    st.error("Could not import split_and_save from data_splitter.py")

# Octagon and Alignment CSS
st.markdown("""
<style>
/* Base Octagon Shape applied to bordered containers */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.octo-marker) {
    clip-path: polygon(5% 0%, 95% 0%, 100% 5%, 100% 95%, 95% 100%, 5% 100%, 0% 95%, 0% 5%);
    transition: all 0.6s ease;
    padding: 20px;
    border: none !important;
}

/* Stagger (Askew) Alignments */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.octo-pos-1) { margin-right: 15% !important; margin-left: 0% !important; margin-top: 10px; }
div[data-testid="stVerticalBlockBorderWrapper"]:has(.octo-pos-2) { margin-left: 15% !important; margin-right: 0% !important; margin-top: -20px; }
div[data-testid="stVerticalBlockBorderWrapper"]:has(.octo-pos-3) { margin-right: 15% !important; margin-left: 0% !important; margin-top: -20px; }

/* Octagon Lighting States */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.octo-state-pending) {
    background: #422a14 !important; /* Dimmed dark orange/brown */
    opacity: 0.5;
    pointer-events: none;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.octo-state-active) {
    background: #e65100 !important; /* Bright glowing orange */
    box-shadow: inset 0 0 20px rgba(255,150,0,0.8);
    opacity: 1;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.octo-state-complete) {
    background: #ff9800 !important; /* Stable completed orange */
    opacity: 0.95;
}

/* Make text inside the colored octagons readable */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.octo-marker) * {
    color: white !important;
}

/* Animated GO Button */
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
    width: 100%;
}
.stButton>button[kind="primary"]:hover { background-color: #45a049 !important; }

/* Hide marker elements */
.octo-marker { display: none; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Auto-Model Setup Pipeline")

if 'step' not in st.session_state: st.session_state.step = 0
if 'df' not in st.session_state: st.session_state.df = None
if 'df_clean' not in st.session_state: st.session_state.df_clean = None

# ---------------------------------------------------------
# SECTION 1: Data Setup (Matches Sketch)
# ---------------------------------------------------------
sec1 = st.container(border=True)
with sec1:
    state1 = 'active' if st.session_state.step == 0 else 'complete'
    st.markdown(f'<div class="octo-marker octo-pos-1 octo-state-{state1}"></div>', unsafe_allow_html=True)
    
    st.markdown("## SECTION 1: DATA SETUP")
    uploaded_file = st.file_uploader("Drop a .csv, .xlsx, or .json file here", type=['csv', 'xlsx', 'json'])
    
    if uploaded_file is not None:
        if st.session_state.get('curr_file') != uploaded_file.name:
            if uploaded_file.name.endswith('.csv'): st.session_state.df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'): st.session_state.df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'): st.session_state.df = pd.read_json(uploaded_file)
            st.session_state.curr_file = uploaded_file.name
            st.session_state.step = 0
    else:
        if st.session_state.get('curr_file') != 'default_train':
            default_path = os.path.join(os.path.dirname(__file__), 'train.csv')
            if os.path.exists(default_path):
                st.session_state.df = pd.read_csv(default_path)
                st.session_state.curr_file = 'default_train'

    if st.session_state.df is not None:
        df = st.session_state.df
        st.write("---")
        st.write("### Data Preview | Details")
        st.dataframe(df.head(4), use_container_width=True)
        
        # Implement sketch feature: Empty cells logic
        st.write("#### Data Quality (Empty Cells)")
        working_df = df.copy()
        
        missing_sums = working_df.isna().sum()
        cols_with_missing = missing_sums[missing_sums > 0].index.tolist()
        
        if not cols_with_missing:
            st.write("✅ No missing values detected in dataset.")
        else:
            for col in missing_sums.index:
                nan_count = missing_sums[col]
                if nan_count > 0:
                    c1, c2 = st.columns([2, 1])
                    c1.write(f"`{col}`: {nan_count} NaNs")
                    # Default is Drop Row if NaNs exist
                    action = c2.selectbox(f"Action for {col}", ["Do nothing", "Drop row", "Drop column"], index=1, key=f"action_{col}")
                    if action == "Drop row":
                        working_df = working_df.dropna(subset=[col])
                    elif action == "Drop column":
                        working_df = working_df.drop(columns=[col])

        # Categorical data raw output
        st.write("#### Categorical Summary")
        cat_cols = working_df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            c1, c2 = st.columns([1, 4])
            c1.write("✅ Detected")
            c2.write(f"Columns: {', '.join(cat_cols)}")
        else:
            st.write("No categorical columns detected.")
            
        st.write("---")
        if st.button("ACCEPT DATA SETTINGS", type="primary" if st.session_state.step == 0 else "secondary"):
            st.session_state.df_clean = working_df
            st.session_state.step = max(st.session_state.step, 1)
            st.rerun()

# ---------------------------------------------------------
# SECTION 2: Splitting & Configuration
# ---------------------------------------------------------
sec2 = st.container(border=True)
with sec2:
    state2 = 'pending' if st.session_state.step < 1 else ('active' if st.session_state.step == 1 else 'complete')
    st.markdown(f'<div class="octo-marker octo-pos-2 octo-state-{state2}"></div>', unsafe_allow_html=True)
    
    st.markdown("## SECTION 2: CONFIGURATION & SPLIT")
    if st.session_state.df_clean is not None and st.session_state.step >= 1:
        df_clean = st.session_state.df_clean
        
        target_options = list(df_clean.columns)
        default_target_idx = len(target_options) - 1
        if 'target' in target_options: default_target_idx = target_options.index('target')
            
        c1, c2 = st.columns(2)
        target_col = c1.selectbox("Select Target Column", target_options, index=default_target_idx)
        
        auto_task_idx = 0
        if df_clean[target_col].dtype == 'object' or df_clean[target_col].nunique() < 20: 
            auto_task_idx = 1
        task_type = c2.radio("Select Task Type", ["regression", "classification"], index=auto_task_idx)

        # Feature Significance execution exactly when Target is set
        try:
            from sklearn.feature_selection import f_classif, f_regression
            df_encoded = df_clean.copy()
            for c in df_encoded.columns:
                if df_encoded[c].dtype == 'object':
                    df_encoded[c] = df_encoded[c].astype('category').cat.codes
                    
            X_sig = df_encoded.drop(columns=[target_col]).fillna(0)
            y_sig = df_encoded[target_col].fillna(0)
            
            F, pval = f_classif(X_sig, y_sig) if task_type == 'classification' else f_regression(X_sig, y_sig)
            
            sig_df = pd.DataFrame({"Feature": X_sig.columns, "F-Value": F}).sort_values('F-Value', ascending=False)
            st.write("**Feature Significance Prediction:**")
            st.dataframe(sig_df.style.background_gradient(subset=['F-Value'], cmap='viridis'), use_container_width=True)
        except Exception:
            pass

        st.write("---")
        st.write("### Data Splitter Configuration")
        col3, col4 = st.columns(2)
        test_size = col3.slider("Test Size Proportion", 0.05, 0.5, 0.2, 0.05)
        valid_size = col4.slider("Validation Size Proportion", 0.05, 0.5, 0.2, 0.05)
        
        # Model Configuration
        st.write("### Model Options")
        models_dir = os.path.join(os.path.dirname(__file__), 'Models')
        available_files = [f for f in os.listdir(models_dir) if f.endswith('.py')]
        valid_models = [m.replace('.py', '') for m in available_files if ('Regressor' if task_type == 'regression' else 'Classifier') in m or 'SVC' in m]
        
        selected_models = st.multiselect("Models to Execute", valid_models, default=valid_models[:3] if len(valid_models) >= 3 else valid_models)

        if st.button("GO: EXECUTE AND EVALUATE", type="primary" if st.session_state.step == 1 else "secondary"):
            outputs_dir = os.path.join(os.path.dirname(__file__), 'temp_splits')
            os.makedirs(outputs_dir, exist_ok=True)
            try:
                train_p, test_p, valid_p, train_l, test_l, valid_l = split_and_save(
                    df_clean, target_col, test_size, valid_size, outputs_dir, task_type
                )
                st.session_state.data_files = {'train': train_p, 'test': test_p, 'valid': valid_p}
                st.session_state.selected_models = selected_models
                st.session_state.step = 2
                st.rerun()
            except Exception as e:
                st.error(f"Error during splitting: {e}")

# ---------------------------------------------------------
# SECTION 3: Execution & Results
# ---------------------------------------------------------
sec3 = st.container(border=True)
with sec3:
    state3 = 'pending' if st.session_state.step < 2 else 'active'
    st.markdown(f'<div class="octo-marker octo-pos-3 octo-state-{state3}"></div>', unsafe_allow_html=True)
    
    st.markdown("## SECTION 3: EVALUATION REPORT")
    if st.session_state.step >= 2:
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, model_name in enumerate(st.session_state.selected_models):
            status_text.text(f"Evaluating {model_name}...")
            try:
                module = importlib.import_module(model_name)
                class_name = model_name.replace('_', '')
                if hasattr(module, class_name):
                    evaluator_class = getattr(module, class_name)
                    default_hyperparams = {'fit_intercept': [True, False]} if 'Linear' in class_name else {'n_estimators': [10, 20]} if 'RandomForest' in class_name else {}
                    evaluator = evaluator_class(st.session_state.data_files, default_hyperparams)
                    best_model, params, score = evaluator.tune_and_evaluate()
                    results.append({"Model": model_name.replace('_Eval', ''), "Score": score})
            except Exception as e:
                st.error(f"Error evaluating {model_name}: {e}")
                
            progress_bar.progress((i + 1) / len(st.session_state.selected_models))
            
        status_text.text("Evaluation Complete!")
        
        if results:
            results_df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
            st.dataframe(results_df, use_container_width=True)
            st.bar_chart(results_df.set_index('Model')['Score'])
            
        if st.button("RESET PIPELINE", key="reset"):
            st.session_state.clear()
            st.rerun()
