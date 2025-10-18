import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import altair as alt
import tempfile
import re



# Load YOLO model
model_path = r'C:\Users\Syed\Desktop\dataset\trained_model\rice_leaf_yolov12\weights\best.pt'
model = YOLO(model_path)


# Disease info and remedies
disease_info = {
    "Bacterial leaf blight": {
        "description": (
            "Bacterial leaf blight (BLB) is caused by the bacterium *Xanthomonas oryzae pv. oryzae*. "
            "It commonly occurs in warm, humid climates, especially in poorly drained fields or after heavy rainfall. "
            "Symptoms include yellowing of leaf tips, wilting, V-shaped lesions starting from leaf edges, "
            "and stunted growth. Severe infection can significantly reduce yield."
        ),
        "remedy": (
            "Use copper-based bactericides or streptomycin sprays. "
            "Maintain proper water management, avoid excess nitrogen fertilizer, "
            "and plant resistant rice varieties if available."
        )
    },
    "Brown spot": {
        "description": (
            "Brown spot is a fungal disease caused by *Bipolaris oryzae*. "
            "It often appears under conditions of low soil fertility or drought stress. "
            "Symptoms include small brown lesions on leaves and grains, which may coalesce "
            "and reduce photosynthesis. Severe infection can lead to poor grain filling."
        ),
        "remedy": (
            "Apply fungicides such as triazoles. "
            "Remove infected leaves, improve soil fertility, and ensure proper irrigation."
        )
    },
    "Hispa": {
        "description": (
            "Hispa is an insect pest (*Dicladispa armigera*) that feeds on rice leaves. "
            "It is more common during the early vegetative stages in warm, humid conditions. "
            "Symptoms include silver streaks on leaves, leaf skeletonization, and reduced photosynthetic area."
        ),
        "remedy": (
            "Use appropriate insecticides or neem-based sprays. "
            "Regular field monitoring and removal of heavily infested leaves can reduce spread."
        )
    },
    "Leaf smut": {
        "description": (
            "Leaf smut is caused by the fungus *Entyloma oryzae*. "
            "It usually appears in humid environments and spreads via water splashes. "
            "Symptoms include elongated black streaks or smut pustules on leaves, which can reduce leaf efficiency."
        ),
        "remedy": (
            "Apply systemic fungicides, remove affected leaves, and rotate crops. "
            "Maintain good field sanitation to prevent spread."
        )
    },
    "LeafBlast": {
        "description": (
            "Leaf blast is caused by the fungus *Magnaporthe oryzae*. "
            "It is one of the most destructive rice diseases, appearing during cool, wet conditions. "
            "Symptoms include diamond-shaped lesions with gray centers and brown edges, "
            "which can coalesce and kill the leaves."
        ),
        "remedy": (
            "Apply recommended fungicides early in the season. "
            "Ensure proper plant spacing, avoid dense planting, and use resistant varieties when possible."
        )
    },
    "Healthy": {
        "description": (
            "The leaf is healthy with no visible disease symptoms. "
            "It appears green, vibrant, and fully turgid with normal growth patterns. "
            "Maintain good agronomic practices to prevent future infections."
        ),
        "remedy": "No action needed. Continue standard cultivation practices."
    }
}


# Streamlit UI
st.set_page_config(
    page_title="Rice Leaf Disease Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page title
st.title("🌾 Rice Leaf Disease Detection System")
st.write("---")

# Upload section
st.header("Upload Rice Leaf Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(uploaded_file.read())
    temp_file.close()

    # Display uploaded image
    image = Image.open(temp_file.name)
    
    col1, col2, col3 = st.columns([1, 2, 1])  # middle column is wider
    with col2:
        st.image(image, caption="Uploaded Image", width=600)

    
    # Make prediction
    results = model.predict(source=temp_file.name)
    result = results[0]

    if len(result.boxes) > 0:
        box = result.boxes[0]
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = model.names[cls_id]
    else:
        cls_name = "Healthy"
        conf = 1.0

    info = disease_info.get(cls_name, {"description": "N/A", "remedy": "N/A"})

    
    # Disease Info & Prediction Section
    st.markdown(f"""
    <div style='background-color:#f0f8ff; padding:20px; border-radius:10px;'>
        <h1 style='color:#2E8B57; font-size:46px; margin-bottom:10px;'>Predicted Disease: {cls_name}</h1>
        <h2 style='color:#333333; font-size:28px; margin-bottom:15px;'>Confidence: {conf*100:.2f}%</h2>
        <p style='font-size:24px; line-height:1.6; text-align:justify; color:#000000; margin-bottom:15px;'>
        <b>Description:</b> {info['description'].replace('*', '')}</p>
        <p style='font-size:24px; line-height:1.6; text-align:justify; color:#8B0000; margin-bottom:10px;'><b>Remedy:</b> {info['remedy']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("") 
    st.write("") 
    
    
    # 🌿 Treatment Planner Section
    st.header("🌿 Treatment Planner")

    # Base treatment data per acre
    treatment_data = {
        "Bacterial leaf blight": {"chemical": "Copper-based bactericide", "quantity_per_acre": 500, "unit": "ml"},
        "Brown spot": {"chemical": "Triazole fungicide", "quantity_per_acre": 400, "unit": "ml"},
        "Hispa": {"chemical": "Neem-based insecticide", "quantity_per_acre": 1000, "unit": "ml"},
        "Leaf smut": {"chemical": "Systemic fungicide", "quantity_per_acre": 350, "unit": "ml"},
        "LeafBlast": {"chemical": "Recommended fungicide", "quantity_per_acre": 500, "unit": "ml"},
        "Healthy": {"chemical": "None", "quantity_per_acre": 0, "unit": ""}
    }

    # Detailed treatment steps
    detailed_steps = {
        "Bacterial leaf blight": [
            "Mix 500 ml of copper-based bactericide in 200 litres of clean water",
            "Spray the solution uniformly on both sides of the leaves using a knapsack sprayer",
            "Repeat the spray after 7 days if symptoms persist",
            "Avoid applying excess nitrogen fertilizer during this period",
            "Maintain proper drainage to prevent standing water"
        ],
        "Brown spot": [
            "Mix 400 ml of triazole fungicide in 200 litres of water",
            "Spray uniformly during early infection stage, focusing on lower leaves",
            "Remove and destroy infected leaves to reduce spore spread",
            "Apply balanced fertilization; avoid excess nitrogen which favors disease",
            "Ensure proper field drainage to prevent water stagnation",
            "Monitor field regularly and reapply fungicide after 10–14 days if symptoms persist"
        ],
        "Hispa": [
            "Mix 1000 ml of neem-based insecticide in 250 litres of water per the required acreage",
            "Spray affected leaves thoroughly, covering both upper and lower surfaces",
            "Repeat the spray after 10 days if infestation continues",
            "Clip and destroy heavily infested leaves to reduce the beetle population",
            "Avoid excessive nitrogen fertilizer application as it encourages pest multiplication",
            "Maintain proper field sanitation — remove nearby weeds and grasses that serve as alternate hosts",
            "In early stages, handpick and crush adult beetles during morning hours to reduce their spread",
            "Ensure proper water management — keep the field drained for a few days after spraying to reduce breeding sites",
        ],

        "LeafSmut": [
            "Mix 350 ml of systemic fungicide in 200 litres of water",
            "Spray evenly on both leaf surfaces",
            "Remove affected leaves and maintain sanitation",
            "Avoid excessive nitrogenous fertilizers",
            "Use resistant or tolerant rice varieties",
            "Ensure proper field drainage to reduce humidity"
        ],
        "LeafBlast": [
            "Mix 500 ml of recommended fungicide in 200 litres of water",
            "Spray at the first appearance of symptoms",
            "Ensure proper plant spacing to avoid dense planting",
            "Remove and destroy infected plant debris",
            "Use blast-resistant rice varieties",
            "Apply balanced fertilization; avoid excess nitrogen"
        ]
    }

    cls_name_normalized = cls_name.strip()
    # Input Css
    st.markdown(
        """
        <style>
        /* Target the number input box */
        div[data-baseweb="input"] input {
            font-size: 30px;       /* Increase text size inside the box */
            height: 60px;          /* Increase height of the box */
            
        }

        /* Target the label above the input */
        label[data-testid="stWidgetLabel"] p {
            font-size: 26px;       /* Increase label size */
            font-weight: bold;
            color: #F227F5
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Get input From User
    acres = st.number_input(
        "Enter the number of acres of your rice field 🌾",
        min_value=0,
        step=1
    )

    

    # Helper function to scale numeric values in step text
    def scale_step_text(step_text: str, acres: float) -> str:
        def repl(match):
            num = float(match.group(1))
            unit = match.group(2)
            total = num * acres
            total_str = str(int(total)) if total.is_integer() else f"{total:.2f}"
            return f"{total_str} {unit}"
        return re.sub(r'(\d+(?:\.\d+)?)\s*(ml|litres|liter|litre|kg|g)', repl, step_text)

    if acres > 0:
        treatment = treatment_data.get(cls_name_normalized, {"chemical": "N/A", "quantity_per_acre": 0, "unit": ""})
        total_quantity = treatment["quantity_per_acre"] * acres

        if cls_name_normalized != "Healthy" and treatment["quantity_per_acre"] > 0:
            base_steps = detailed_steps.get(cls_name_normalized, [])
            scaled_steps = [scale_step_text(s, acres) for s in base_steps]
            steps_html = f"""
            <ul style='padding-left: 80px;'>
                {''.join([f"<li style='margin-bottom:6px; color:#8B0000; font-size:24px;'>{s}.</li>" for s in scaled_steps])}
            </ul>
            """


            st.markdown(f"""
            <div style='background-color:#E8F5E9; padding:20px; border-radius:10px;'>
                <h3 style='color:#2E8B57;font-size:46px'>🧪 Recommended Treatment for {acres} acres</h3>
                <p style='font-size:24px; color:#000000; padding-left:80px;'>
                    <b>Chemical:</b> {treatment['chemical']}<br>
                    <b>Required Quantity:</b> {total_quantity:.2f} {treatment['unit']}
                </p>
                <h4 style='color:#2E7D32; margin-top:15px;font-size:40px'>📝 Detailed Treatment Steps:</h4>
                {steps_html}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("✅ Your field is healthy. No treatment is required. Continue good practices.")


    st.write("") 
    st.write("")
    
    # Prediction Table
    st.header("Prediction Table")

    df_table = pd.DataFrame({
        "Predicted Disease": [cls_name],
        "Confidence (%)": [f"{conf * 100:.2f}%"]
    })

    # Apply styling to align left
    st.dataframe(
        df_table.style.set_properties(**{'text-align': 'left'}),
        width=800,
        height=70
    )

    st.write("") 
    st.write("")
    
    
    # Confidence Bar Chart
    st.header("Confidence Bar Chart")

    df_bar = pd.DataFrame({
        "Disease": [cls_name],
        "Confidence": [conf*100]
    })

    bar_chart = alt.Chart(df_bar).mark_bar(color="#DC1010").encode(
        x="Disease",
        y="Confidence",
        tooltip=["Disease", "Confidence"]
    ).properties(width=800, height=300) 
    st.altair_chart(bar_chart)  

    st.write("") 
    st.write("")
    

    # Prediction Pie Chart
    st.header("Prediction Pie Chart")
    df_pie = pd.DataFrame({
        'Category': [cls_name, 'Others'],
        'Percentage': [conf*100, 100 - conf*100]
    })

    st.write("") 
    st.write("")   
    
    pie_chart = alt.Chart(df_pie).mark_arc().encode(
        theta=alt.Theta(field="Percentage", type="quantitative"),
        color=alt.Color(field="Category", type="nominal", scale=alt.Scale(range=["#E662B8", "#D3D3D3"])),
        tooltip=['Category', 'Percentage']
    ).properties(height=300)
    st.altair_chart(pie_chart, use_container_width=True)
    
    st.write("")   
    st.write("")
    
    # Author Details
    st.markdown(
            """
            <p style="font-size:20px; margin-top:0;">
                👨‍💻 Developed by: <b>KANNIGA PARAMESHWARI R G</b>
            </p>
            <p style="font-size:20px;">💡 Idea: <i>Rice Leaf Disease Detection System</i></p>
            <p style="font-size:20px;">🛠️ Tech Stack: Python, PyTorch, YOLOv12 (Ultralytics), OpenCV, Streamlit, NumPy, Pandas</p>
            """,
            unsafe_allow_html=True
        )
