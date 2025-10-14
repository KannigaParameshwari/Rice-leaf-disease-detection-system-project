import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import altair as alt
import tempfile

# ----------------------------
# Load YOLO model
# ----------------------------
model_path = r'C:\Users\Syed\Desktop\dataset\trained_model\rice_leaf_yolov12\weights\best.pt'
model = YOLO(model_path)

# ----------------------------
# Disease info and remedies
# ----------------------------
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

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(
    page_title="Rice Leaf Disease Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page title
st.title("🌾 Rice Leaf Disease Detection System")
st.write("---")

# Upload section
st.subheader("Upload Rice Leaf Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(uploaded_file.read())
    temp_file.close()

    # Display uploaded image
    image = Image.open(temp_file.name)
    # st.image(image, caption="Uploaded Image", width=600 )
    
    col1, col2, col3 = st.columns([1, 2, 1])  # middle column is wider
    with col2:
        st.image(image, caption="Uploaded Image", width=600)

    # ----------------------------
    # Make prediction
    # ----------------------------
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

    # ----------------------------
    # Disease Info & Prediction Section
    # ----------------------------
    st.markdown(f"""
    <div style='background-color:#f0f8ff; padding:20px; border-radius:10px;'>
        <h1 style='color:#2E8B57; font-size:46px; margin-bottom:10px;'>Predicted Disease: {cls_name}</h1>
        <h2 style='color:#333333; font-size:28px; margin-bottom:15px;'>Confidence: {conf*100:.2f}%</h2>
        <p style='font-size:24px; line-height:1.6; text-align:justify; color:#000000; margin-bottom:15px;'>
        <b>Description:</b> {info['description'].replace('*', '')}
    </p>
        <p style='font-size:24px; line-height:1.6; text-align:justify; color:#8B0000; margin-bottom:10px;'><b>Remedy:</b> {info['remedy']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Prediction Table")

    # Convert confidence to string so it aligns left
    df_table = pd.DataFrame({
        "Predicted Disease": [cls_name],
        "Confidence (%)": [f"{conf * 100:.2f}%"]
    })

    # Apply styling to align left
    st.dataframe(
        df_table.style.set_properties(**{'text-align': 'left'}),
        width=800,
        height=100
    )


    # ----------------------------
    # Confidence Bar Chart
    # ----------------------------
    st.subheader("Confidence Bar Chart")

    df_bar = pd.DataFrame({
        "Disease": [cls_name],
        "Confidence": [conf*100]
    })

    bar_chart = alt.Chart(df_bar).mark_bar(color="#DC1010").encode(
        x="Disease",
        y="Confidence",
        tooltip=["Disease", "Confidence"]
    ).properties(width=800, height=300)  # <-- set width manually

    st.altair_chart(bar_chart)  # do NOT use_container_width


    # ----------------------------
    # Prediction Pie Chart
    # ----------------------------
    st.subheader("Prediction Pie Chart")
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
