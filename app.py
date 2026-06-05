import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import io

st.set_page_config(page_title="Defect Detection", layout="wide")

st.title("🔬 Defect Detection System")
st.subheader("Manufacturing Quality Control - CIPET Chennai")

# Input method
option = st.radio("Choose Input Method:", ["📁 Upload Image", "📷 Use Camera"])

image = None

if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload Component Image", type=["jpg","jpeg","png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
else:
    camera_photo = st.camera_input("Take Photo of Component")
    if camera_photo:
        file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if image is not None:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Input Image", use_column_width=True)

    st.sidebar.header("⚙ GD&T Settings")
    max_size = st.sidebar.slider("Max Defect Size (px)", 10, 500, 100)
    max_count = st.sidebar.slider("Max Defect Count", 1, 50, 10)
    max_area = st.sidebar.slider("Max Defect Area %", 0.1, 10.0, 2.0)

    if st.button("▶ RUN INSPECTION", use_container_width=True):
        with st.spinner("Scanning..."):

            # OpenCV Detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5,5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            critical = major = minor = 0
            output = image_rgb.copy()
            gdt_data = []
            h_img, w_img = image.shape[:2]

            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)

                # Location detection
                cx = x + w//2
                cy = y + h//2
                if cx < w_img//3:
                    loc_x = "Left"
                elif cx < 2*w_img//3:
                    loc_x = "Center"
                else:
                    loc_x = "Right"

                if cy < h_img//3:
                    loc_y = "Top"
                elif cy < 2*h_img//3:
                    loc_y = "Middle"
                else:
                    loc_y = "Bottom"

                location = f"{loc_y}-{loc_x}"

                # Shape detection
                aspect_ratio = w / h if h > 0 else 1
                circularity = area / (w * h) if w * h > 0 else 0

                if aspect_ratio > 3:
                    shape = "Linear Crack"
                elif circularity > 0.7:
                    shape = "Circular Pit"
                elif aspect_ratio < 0.3:
                    shape = "Vertical Crack"
                else:
                    shape = "Irregular"

                # Severity
                if area > 1000:
                    critical += 1
                    color = (255, 0, 0)
                    label = "Critical"
                elif area > 300:
                    major += 1
                    color = (255, 165, 0)
                    label = "Major"
                elif area > 20:
                    minor += 1
                    color = (255, 255, 0)
                    label = "Minor"
                else:
                    continue

                tol = "OUT ❌" if area > max_size else "OK ✅"
                cv2.rectangle(output, (x,y), (x+w,y+h), color, 2)
                cv2.putText(output, label, (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                gdt_data.append({
                    "Defect #": i+1,
                    "Severity": label,
                    "Shape": shape,
                    "Location": location,
                    "Size (px)": round(area),
                    "W x H": f"{w}x{h}",
                    "GD&T": tol
                })

            total = critical + major + minor
            img_area = image.shape[0] * image.shape[1]
            defect_pct = (sum(cv2.contourArea(c) for c in contours
                         if cv2.contourArea(c) > 20) / img_area) * 100

            # Show defect map
            st.image(output, caption="Defect Map", use_column_width=True)

            # Verdict
            if critical > 0 or total > max_count or defect_pct > max_area:
                verdict = "❌ FAIL"
                st.error(f"VERDICT: {verdict}")
            elif major > 0:
                verdict = "⚠ WARNING"
                st.warning(f"VERDICT: {verdict}")
            else:
                verdict = "✅ PASS"
                st.success(f"VERDICT: {verdict}")

            # Metrics
            st.subheader("📊 Inspection Report")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Defects", total)
            col2.metric("🔴 Critical", critical)
            col3.metric("🟠 Major", major)
            col4.metric("🟡 Minor", minor)

            col1, col2 = st.columns(2)
            col1.metric("Defect Area %", f"{defect_pct:.2f}%")
            col2.metric("GD&T Limit %", f"{max_area}%")

            if defect_pct > max_area:
                st.error("Surface OUT OF TOLERANCE ❌")
            else:
                st.success("Surface WITHIN TOLERANCE ✅")

            # Defect table
            if gdt_data:
                st.subheader("📐 GD&T Defect Table")
                st.table(gdt_data)

            # Smart Recommendations
            st.subheader("💡 Recommendations")

            if critical > 0:
                st.error("🔴 CRITICAL DEFECTS FOUND")
                st.write("→ Reject part immediately — do not use")
                st.write("→ Check mold/die condition")
                st.write("→ Review pouring temperature")
                st.write("→ Check for moisture in raw material")
                st.write("→ Inspect cooling rate")

            if major > 0:
                st.warning("🟠 MAJOR DEFECTS FOUND")
                st.write("→ Send part for rework")
                st.write("→ Check machine parameters")
                st.write("→ Review operator technique")
                st.write("→ Increase inspection frequency")

            if minor > 0:
                st.info("🟡 MINOR DEFECTS FOUND")
                st.write("→ Part may be acceptable")
                st.write("→ Monitor surface finish process")
                st.write("→ Check tool condition")

            if total == 0:
                st.success("✅ No defects — excellent quality!")
                st.write("→ Document for quality records")
                st.write("→ Continue current process parameters")

            # Shape based recommendations
            shapes = [d["Shape"] for d in gdt_data]
            if "Linear Crack" in shapes or "Vertical Crack" in shapes:
                st.warning("⚠ Cracks detected!")
                st.write("→ Check cooling rate — too fast cooling causes cracks")
                st.write("→ Review material composition")
                st.write("→ Check residual stress in part")

            if "Circular Pit" in shapes:
                st.warning("⚠ Porosity/Pits detected!")
                st.write("→ Check for gas entrapment during casting")
                st.write("→ Review shielding gas in welding")
                st.write("→ Check moisture in mold material")

            # PDF Report
            st.subheader("📄 Download Report")
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            report_text = f"""
DEFECT INSPECTION REPORT
========================
Date & Time    : {now}
Institution    : CIPET Chennai

SUMMARY
-------
Total Defects  : {total}
Critical       : {critical}
Major          : {major}
Minor          : {minor}
Defect Area    : {defect_pct:.2f}%
Verdict        : {verdict}

GD&T ANALYSIS
-------------
Max Defect Size  : {max_size} px
Max Defect Count : {max_count}
Max Area %       : {max_area}%
Status           : {'OUT OF TOLERANCE' if defect_pct > max_area else 'WITHIN TOLERANCE'}

DEFECT DETAILS
--------------
"""
            for d in gdt_data:
                report_text += f"""
Defect #{d['Defect #']}
  Severity : {d['Severity']}
  Shape    : {d['Shape']}
  Location : {d['Location']}
  Size     : {d['Size (px)']} px
  GD&T     : {d['GD&T']}
"""

            report_text += """
RECOMMENDATIONS
---------------
"""
            if critical > 0:
                report_text += "- Reject part immediately\n"
                report_text += "- Check mold/die condition\n"
                report_text += "- Review pouring temperature\n"
            elif major > 0:
                report_text += "- Send for rework\n"
                report_text += "- Check machine parameters\n"
            else:
                report_text += "- Part acceptable for use\n"
                report_text += "- Continue current process\n"

            st.download_button(
                label="📥 Download Report as TXT",
                data=report_text,
                file_name=f"inspection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )