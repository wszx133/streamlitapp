import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import io
import easychart
from PIL import Image

from filesplit.merge import Merge
import os

path = os.getcwd()

if os.path.exists(path+"\\stacking_model") and not os.path.exists(path+"\\stacking_model\\stacking_model1.pkl"):
    merge = Merge(path+"\\stacking_model", path+"\\", "stacking_model.pkl")
    merge.merge()
    print("合并")
else:
    pass

easychart.config.rendering.responsive = True

st.set_page_config(
    page_title="IVIG Resistance Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Outcome mapping
OUTCOME_LABELS = {
    0: "None",
    1: "Mild",
    2: "Moderate",
    3: "Severe"
}

# Custom colors as specified
COLORS = ['#66C2A5', '#FFC107', '#E67E22', '#C0392B']  # Corresponding to None, Mild, Moderate, Severe


# ================== Model Caching ==================
@st.cache_resource
def load_model_scaler_background():
    try:
        model = joblib.load(r'stacking_model.pkl')
        scaler = joblib.load(r'scaler.pkl')

        try:
            background_scaled = np.load(r'shap_background.npy')
            #st.success(f"✅ Loaded successfully: Model + scaler + SHAP background ({len(background_scaled)} samples)")
        except FileNotFoundError:
            st.warning("⚠️ shap_background.npy not found, using random data instead")
            background_scaled = np.random.rand(100, 8)

        if not hasattr(model, "predict_proba"):
            st.error("❌ Model does not support probability prediction")
            st.stop()
        return model, scaler, background_scaled

    except Exception as e:
        st.error(f"❌ Loading failed: {str(e)}")
        st.stop()


# Load resources
model, scaler, background_scaled = load_model_scaler_background()


# ================== Input Form with Horizontal Feature Table ==================
def create_input_form():
    with st.sidebar:
        st.markdown('<h3 class="title">📋 Patient Feature Input</h3><br>', unsafe_allow_html=True)
        input_config = {
            'Male': {'label': "Gender (Male)", 'type': 'binary', 'options': ["No", "Yes"]},
            'CRP': {'label': "CRP", 'type': 'numeric', 'min': 0.01, 'max': 366.68, 'default': 50.0, 'step': 0.01},
            'PLT': {'label': "PLT", 'type': 'numeric', 'min': 39.2, 'max': 987.0, 'default': 300.0, 'step': 1.0},
            'ALB': {'label': "ALB", 'type': 'numeric', 'min': 23.1, 'max': 97.0, 'default': 40.0, 'step': 0.1},
            'TBIL': {'label': "TBIL", 'type': 'numeric', 'min': 0.76, 'max': 113.9, 'default': 15.0, 'step': 0.1},
            'PLR': {'label': "PLR", 'type': 'numeric', 'min': 4.25, 'max': 1350.0, 'default': 150.0, 'step': 1.0},
            'CALLY': {'label': "CALLY", 'type': 'numeric', 'min': 0.013, 'max': 18427.4, 'default': 1000.0,
                      'step': 1.0},
            'IVIG resistant': {'label': "IVIG resistant", 'type': 'binary', 'options': ["No", "Yes"]}
        }

        inputs = {}
        for feat, config in input_config.items():
            if config['type'] == 'binary':
                val = st.selectbox(config['label'], config['options'], key=f"input_{feat}", index=0)
                inputs[feat] = 1 if val == "Yes" else 0
            else:
                inputs[feat] = st.slider(
                    config['label'],
                    min_value=float(config['min']),
                    max_value=float(config['max']),
                    value=float(config['default']),
                    step=float(config.get('step', 1.0)),
                    key=f"input_{feat}"
                )

        # Prepare feature data for horizontal table
        feature_data = {}
        for i, (feat, value) in enumerate(inputs.items()):
            # Convert binary features to text descriptions
            if feat == 'Male':
                display_value = "Yes" if value == 1 else "No"
            elif feat == 'IVIG resistant':
                display_value = "Yes" if value == 1 else "No"
            else:
                display_value = round(value, 2) if isinstance(value, float) else value

            feature_data[f"Feature {i + 1}"] = input_config[feat]['label']
            feature_data[f"Value {i + 1}"] = display_value

        # Create horizontal DataFrame (features as columns)
        num_features = len(inputs)
        cols_needed = num_features if num_features % 2 == 0 else num_features + 1  # Ensure even number
        horizontal_data = {}

        # Organize features into columns (2 features per column pair)
        for i in range(0, cols_needed, 2):
            if i < num_features:
                feat_idx = list(inputs.keys())[i]
                horizontal_data[input_config[feat_idx]['label']] = [
                    round(inputs[feat_idx], 2) if isinstance(inputs[feat_idx], float) else
                    "Yes" if inputs[feat_idx] == 1 else "No"
                ]
            if i + 1 < num_features:
                feat_idx = list(inputs.keys())[i + 1]
                horizontal_data[input_config[feat_idx]['label']] = [
                    round(inputs[feat_idx], 2) if isinstance(inputs[feat_idx], float) else
                    "Yes" if inputs[feat_idx] == 1 else "No"
                ]

        input_df = pd.DataFrame(horizontal_data)

        # Return original input for prediction
        original_input_df = pd.DataFrame([inputs], columns=input_config.keys())
        return original_input_df, input_df


# ================== Circular Probability Chart ==================
def plot_probability_ring(probabilities):
    try:
        # Prepare data
        labels = [OUTCOME_LABELS[i] for i in range(4)]
        sizes = probabilities * 100  # Convert to percentages

        # Create circular chart with no gaps
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(aspect="equal"))

        # Create wedges with no gaps
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=COLORS,
            autopct='%1.1f%%', pctdistance=0.85, startangle=140,
            wedgeprops=dict(width=0.3, edgecolor='w', linewidth=0),  # No gap between segments
            textprops=dict(fontsize=12)
        )

        # Improve text appearance
        plt.setp(autotexts, size=10, weight="bold", color="white")

        # Add center circle and title
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)
        ax.set_title("Outcome Probability Distribution", fontsize=16, pad=20)

        # Save image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        img = Image.open(buf)

        return img
    except Exception as e:
        st.error(f"❌ Failed to generate probability chart: {str(e)}")
        return None

def prob_pie(labels, values, title, color):
    chart = easychart.new("pie")
    #chart.subtitle = "Source: American Red Cross"
    chart.tooltip = "{point.percentage:.0f}%"
    chart.plot(values, index=labels, labels="{point.name} ({point.y}%)", innerSize=["60%", "80%"], colors=color)
    chart.legend.enabled = False
    chart.title = title
    chart.title.align = 'center'
    chart.height = 320
    #chart.subtitle.align = 'center'
    st.components.v1.html(easychart.rendering.render(chart), height=320)

# ================== SHAP Force Plots ==================
def plot_shap_force_plots(model, input_df, input_scaled, background_scaled):
    try:
        # Create explainer
        explainer = shap.KernelExplainer(model.predict_proba, background_scaled)
        shap_values = explainer.shap_values(input_scaled)

        # Feature names (English)
        feat_names = ['Gender (Male)', 'CRP', 'PLT', 'ALB', 'TBIL', 'PLR', 'CALLY', 'IVIG resistant']

        # Create force plots for each outcome
        force_plots = []
        for i in range(4):
            # Get SHAP values for single sample
            shap_single = shap_values[i][0]
            base_val = explainer.expected_value[i]

            # Create force plot with larger size
            plt.figure(figsize=(20, 3), dpi=500)
            shap.force_plot(
                base_val, shap_single, input_df.iloc[0].values,
                feature_names=feat_names, matplotlib=True,
                show=False, text_rotation=15, 
            )
            plt.grid(False)

            # Add title
            plt.title(f"SHAP Force Plot - {OUTCOME_LABELS[i]}", fontsize=16)
            plt.tight_layout()

            # Save as image
            # buf = io.BytesIO()
            # plt.savefig(buf, format="png", dpi=450, bbox_inches="tight")
            # plt.close()
            # buf.seek(0)
            # img = Image.open(buf)

            force_plots.append((OUTCOME_LABELS[i], plt.gcf()))

        return force_plots
    except Exception as e:
        st.error(f"❌ Failed to generate SHAP force plots: {str(e)}")
        return []


# ================== Main Program ==================
def main():
    st.markdown('<h1 class="page-title" style="border-bottom: 1px solid black; padding-bottom: 0.5rem;">🏥 Prediction System for Coronary Artery Lesion Severity in Children with KD</h1><br>', unsafe_allow_html=True)
    # st.markdown("""
    # <p style="text-align:center; color:#666; border-bottom: 1px solid black; padding-bottom: 0.5rem;">Machine learning-based CAL classification prediction with feature contribution analysis</p>
    # """, unsafe_allow_html=True)

    input_df, feature_table = create_input_form()
    
    with st.expander("**📊 Current Input Features**", True):
        st.dataframe(feature_table, use_container_width=True)
    
        pre_button = st.button("🖥️ Start predict", type="primary")
        
        if not pre_button:
            st.markdown("<div style='color: red; text-align: center;'>Please input your feature data and click 'Start predict' button to start!</div>", unsafe_allow_html=True)

    if pre_button:
        with st.expander("**🎯 Prediction Result**", True):
            with st.spinner("Predicting, please wait..."):
                try:
                    # Calculation process
                    input_scaled = scaler.transform(input_df)
                    prob = model.predict_proba(input_scaled)[0]
                    pred_class = np.argmax(prob)
                    pred_class_name = f"{OUTCOME_LABELS[pred_class]}"
                    # prob_plot = plot_probability_ring(prob)
                    # Get force plots for所有 outcomes
                    force_plots = plot_shap_force_plots(model, input_df, input_scaled, background_scaled)

                    error = False
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    error = True
            
            if not error:
                # Show prediction result
                st.markdown(f'<div class="card result-card">Most Likely Outcome: {pred_class_name}</div>', unsafe_allow_html=True)

                with st.container():
                    # Display circular chart
                    st.markdown('''<div style="text-align: center; border-bottom: 1px solid gray; font-size: 24px;">Probability Distribution</div>''', unsafe_allow_html=True)
                    # if prob_plot:
                        # col = st.columns([1, 2, 1])
                        # with col[1]:
                        #    st.image(prob_plot, use_container_width=True)
                    
                    color = {'None':'#66C2A5', 'Mild':'#FFC107', 'Moderate':'#E67E22', 'Severe':'#C0392B'}
                    x = [OUTCOME_LABELS[i] for i in range(4)]
                    y = (prob*100).round(2).tolist()
                    prob_pie(x, y, "Outcome Probability Distribution", [color[k] for k in x])

                    # Probability table for reference
                    prob_df = pd.DataFrame({
                        "Outcome": [OUTCOME_LABELS[i] for i in range(4)],
                        "Probability": prob.round(6),
                        "Percentage": [f"{p * 100:.2f}%" for p in prob]
                    })
                    st.table(prob_df.T)

                with st.container():
                    st.markdown('''<br><div style="text-align: center; border-bottom: 1px solid gray; font-size: 24px;">🔍 Feature Contribution Analysis (SHAP Force Plots)</div>''', unsafe_allow_html=True)
                    st.markdown("<div style='color: red; text-align: center;'>Force plots show each feature's impact on outcomes (red = positive impact, blue = negative impact)</div>", unsafe_allow_html=True)
                    
                    labels = [i for i, _ in force_plots]
                    tabs = st.tabs(labels)
                    # Display four force plots in a single column for better visibility
                    for label, plot in force_plots:
                        #st.subheader(f"{label}")
                        tabs[labels.index(label)].pyplot(plot, use_container_width=True, format='png', dpi=400)


# Custom CSS - 调整布局使内容左对齐
self_css_style = """
<style>
body {
    background-color: #f5f7fa;
}
#tabletablecontainer .handsontable thead th .relative {
    text-align: center !important;
    font-weight: bold;
    over-flow: scroll;
}

[data-testid="stTableStyledTable"] td,  [data-testid="stTableStyledTable"] th {
    text-align: center !important;
}

[data-testid="stTableStyledTable"] {
    over-flow: scroll;
}

.index_name {
    width: 200px !important;
}

[scope="row"], .col_heading, .index_name {
    background: #EEF8FE;
    color: black;
    font-weight: bold;
}

.htNumeric {
    text-align: center !important;
}

.st-key-tablecontainer {
    gap: 0rem;
}
.stMainBlockContainer {
    margin-top: -80px;
}
.reportview-container .main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}
.card {
    background-color: white;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.title {
    color: #2c3e50;
    font-weight: 700;
    margin-bottom: 1rem;
    text-align: center;
    border-bottom: 1px solid black;
}
.stAppHeader {
    background: transparent;
}
.result-card {
    background: linear-gradient(135deg, #f0f9ff 0%, #e6f7ff 100%);
    border-left: 4px solid #1890ff;
}
.stButton>button {
    width: 100%;
    background-color: #1890ff;
    color: white;
    border-radius: 6px;
    padding: 0.6rem 0;
    font-weight: 500;
}
.stButton>button:hover {
    background-color: #096dd9;
}
.stSlider, .stRadio {
    /*margin-bottom: 1.2rem;*/
}
.feature-table {
    font-size: 14px;
    border-collapse: separate;
    border-spacing: 0;
    width: 100%;
}
.feature-table th {
    background-color: #f0f2f5;
    color: #1f2329;
    font-weight: 500;
    text-align: center;
}
.feature-table td, .feature-table th {
    padding: 8px 12px;
    border-bottom: 1px solid #e5e6eb;
}
/* 确保标题居中 */
.page-title {
    text-align: center !important;
    padding: 0px !important;
    font-size: 2.4rem !important;
    margin-top: 0.3rem !important;
}
/* 调整布局使内容左对齐 */
.main-content {
    margin-left: 0;
    padding-left: 0;
}
.results-container {
    margin-left: 0;
    padding-left: 0;
}
/* 调整列比例使内容更靠左 */
.column-container {
    margin-left: 0;
    padding-left: 0;
}
/* 调整图像容器左对齐 */
.image-container-left {
    display: flex;
    justify-content: flex-start;
    align-items: center;
    margin: 1rem 0;
}
</style>
"""
st.markdown(self_css_style, unsafe_allow_html=True)

if __name__ == "__main__":

    main()
