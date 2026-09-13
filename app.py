import io
import itertools
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LunarVision AI",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(90,70,180,.20), transparent 28%),
        radial-gradient(circle at 85% 15%, rgba(30,90,190,.16), transparent 30%),
        linear-gradient(135deg,#050611,#0a0d20 50%,#101532);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#070916,#0d1230);
}

.hero {
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(130,110,255,.35);
    background: rgba(25,25,65,.45);
    margin-bottom: 20px;
}

.hero h1 {
    margin: 0;
    font-size: 2.1rem;
}

.subtitle {
    color: #aeb4d8;
    margin-top: 6px;
}

.result-card {
    padding: 14px;
    border-radius: 14px;
    border: 1px solid rgba(120,110,255,.28);
    background: rgba(8,11,28,.75);
    margin-bottom: 10px;
}

.good {
    color: #69e59b;
    font-weight: 700;
}

.medium {
    color: #ffd166;
    font-weight: 700;
}

.bad {
    color: #ff6878;
    font-weight: 700;
}

.muted {
    color: #9ca4ca;
    font-size: .84rem;
}

.image-box {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 360px;
    padding: 10px;
    border-radius: 12px;
    background: rgba(5,7,18,.7);
    border: 1px solid rgba(120,110,255,.20);
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = []

if "chat" not in st.session_state:
    st.session_state.chat = []

if "analysed" not in st.session_state:
    st.session_state.analysed = False


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <h1>🌙 LunarVision AI</h1>
    <div class="subtitle">
        Chandrayaan-2 Multi-Sensor Image Registration
        · OHRC · TMC · IIRS
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Registration Settings")

features = st.sidebar.slider(
    "SIFT features",
    2000,
    15000,
    10000,
    1000
)

lowe_ratio = st.sidebar.slider(
    "Lowe ratio",
    0.65,
    0.90,
    0.80,
    0.01
)

ransac_threshold = st.sidebar.slider(
    "RANSAC threshold",
    2.0,
    12.0,
    5.0,
    0.5
)

max_size = st.sidebar.slider(
    "Processing size",
    1000,
    3000,
    2000,
    100
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### Recommended

**Features:** 10,000  
**Lowe ratio:** 0.80  
**RANSAC:** 5 px  
**Processing:** 2000 px
""")

st.sidebar.markdown("---")

st.sidebar.caption(
    "Strong/Moderate/Weak indicate feature-based "
    "geometric consistency. They are not absolute "
    "geolocation accuracy."
)


# ============================================================
# IMAGE DECODING
# ============================================================

def decode_image(data):

    try:
        array = np.frombuffer(
            data,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR
        )

        if image is not None:
            return image

    except Exception:
        pass

    try:
        pil = Image.open(
            io.BytesIO(data)
        )

        pil = ImageOps.exif_transpose(
            pil
        )

        pil = pil.convert("RGB")

        rgb = np.array(pil)

        return cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR
        )

    except Exception:
        return None


def pil_image(data):

    image = Image.open(
        io.BytesIO(data)
    )

    image = ImageOps.exif_transpose(
        image
    )

    return image.convert("RGB")


# ============================================================
# COMPACT UPLOAD PREVIEW
# ============================================================

def compact_preview(data):

    try:

        image = pil_image(data)

        image.thumbnail(
            (210, 220),
            Image.Resampling.LANCZOS
        )

        canvas = Image.new(
            "RGB",
            (210, 220),
            (8, 10, 24)
        )

        x = (
            210 - image.width
        ) // 2

        y = (
            220 - image.height
        ) // 2

        canvas.paste(
            image,
            (x, y)
        )

        return canvas

    except Exception:

        return None


# ============================================================
# COMPACT RESULT IMAGE
# ============================================================

def compact_result_image(
    image,
    max_width=500,
    max_height=350
):
    """
    IMPORTANT:
    This limits BOTH width and height.

    The image keeps its original aspect ratio.
    Nothing is stretched.
    Nothing is cropped.
    """

    if image is None:
        return None

    # OpenCV BGR -> RGB
    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    pil = Image.fromarray(
        rgb_image
    )

    pil.thumbnail(
        (max_width, max_height),
        Image.Resampling.LANCZOS
    )

    # Fixed compact display area
    canvas = Image.new(
        "RGB",
        (max_width, max_height),
        (7, 9, 22)
    )

    x = (
        max_width - pil.width
    ) // 2

    y = (
        max_height - pil.height
    ) // 2

    canvas.paste(
        pil,
        (x, y)
    )

    return canvas


# ============================================================
# RESIZE FOR PROCESSING
# ============================================================

def resize_image(
    image,
    max_dimension
):

    h, w = image.shape[:2]

    scale = min(
        1.0,
        max_dimension /
        float(max(h, w))
    )

    if scale >= 1:

        return image, 1.0

    new_w = int(
        w * scale
    )

    new_h = int(
        h * scale
    )

    result = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    return result, scale


# ============================================================
# CLAHE PREPROCESSING
# ============================================================

def enhanced_gray(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(
        gray
    )

    return gray


# ============================================================
# ROOTSIFT
# ============================================================

def root_sift(descriptors):

    if descriptors is None:
        return None

    descriptors = descriptors.astype(
        np.float32
    )

    sums = descriptors.sum(
        axis=1,
        keepdims=True
    )

    descriptors = descriptors / (
        sums + 1e-8
    )

    descriptors = np.sqrt(
        descriptors
    )

    return descriptors


# ============================================================
# SIFT
# ============================================================

def detect_sift(
    image,
    nfeatures
):

    gray = enhanced_gray(
        image
    )

    sift = cv2.SIFT_create(
        nfeatures=int(
            nfeatures
        ),
        contrastThreshold=0.008,
        edgeThreshold=12,
        sigma=1.6
    )

    keypoints, descriptors = (
        sift.detectAndCompute(
            gray,
            None
        )
    )

    descriptors = root_sift(
        descriptors
    )

    return keypoints, descriptors


# ============================================================
# ORB FALLBACK
# ============================================================

def detect_orb(image):

    gray = enhanced_gray(
        image
    )

    orb = cv2.ORB_create(
        nfeatures=10000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        patchSize=31,
        fastThreshold=8
    )

    keypoints, descriptors = (
        orb.detectAndCompute(
            gray,
            None
        )
    )

    return keypoints, descriptors


# ============================================================
# SIFT MATCHES
# ============================================================

def sift_matches(
    des1,
    des2,
    ratio
):

    if des1 is None or des2 is None:
        return []

    if len(des1) < 2 or len(des2) < 2:
        return []

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    matches = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    good = []

    for pair in matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < (
            ratio * n.distance
        ):

            good.append(m)

    return good


# ============================================================
# ORB MATCHES
# ============================================================

def orb_matches(
    des1,
    des2,
    ratio
):

    if des1 is None or des2 is None:
        return []

    if len(des1) < 2 or len(des2) < 2:
        return []

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    matches = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    good = []

    for pair in matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < (
            ratio * n.distance
        ):

            good.append(m)

    return good


# ============================================================
# HOMOGRAPHY
# ============================================================

def calculate_homography(
    kp1,
    kp2,
    matches,
    threshold
):

    if len(matches) < 4:

        return None, None, None

    src = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ]).reshape(
        -1,
        1,
        2
    )

    dst = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ]).reshape(
        -1,
        1,
        2
    )

    H, mask = cv2.findHomography(
        src,
        dst,
        cv2.RANSAC,
        threshold,
        maxIters=15000,
        confidence=0.995
    )

    if H is None or mask is None:

        return None, None, None

    mask = mask.ravel().astype(
        bool
    )

    if np.sum(mask) < 4:

        return None, None, None

    projected = cv2.perspectiveTransform(
        src,
        H
    )

    dst_flat = dst.reshape(
        -1,
        2
    )

    projected_flat = projected.reshape(
        -1,
        2
    )

    errors = np.linalg.norm(
        projected_flat - dst_flat,
        axis=1
    )

    inlier_errors = errors[
        mask
    ]

    mean_error = float(
        np.mean(inlier_errors)
    )

    median_error = float(
        np.median(inlier_errors)
    )

    return (
        H,
        mask,
        mean_error,
        median_error
    )


# ============================================================
# QUALITY
# ============================================================

def quality_label(
    inliers,
    matches,
    error
):

    if matches <= 0:
        return "Weak"

    ratio = (
        inliers /
        float(matches)
    )

    if (
        inliers >= 35
        and ratio >= 0.50
        and error <= 4
    ):

        return "Strong"

    if (
        inliers >= 15
        and ratio >= 0.35
        and error <= 7
    ):

        return "Moderate"

    return "Weak"


# ============================================================
# REGISTRATION ENGINE
# ============================================================

def register_images(
    image1,
    image2,
    nfeatures,
    ratio,
    threshold,
    max_dimension
):

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    img1, scale1 = resize_image(
        image1,
        max_dimension
    )

    img2, scale2 = resize_image(
        image2,
        max_dimension
    )

    # --------------------------------------------------------
    # SIFT
    # --------------------------------------------------------

    kp1, des1 = detect_sift(
        img1,
        nfeatures
    )

    kp2, des2 = detect_sift(
        img2,
        nfeatures
    )

    sift_good = sift_matches(
        des1,
        des2,
        ratio
    )

    best_method = "SIFT"

    best_kp1 = kp1
    best_kp2 = kp2
    best_matches = sift_good

    # --------------------------------------------------------
    # ORB fallback
    # --------------------------------------------------------

    if len(sift_good) < 8:

        orb_kp1, orb_des1 = detect_orb(
            img1
        )

        orb_kp2, orb_des2 = detect_orb(
            img2
        )

        orb_good = orb_matches(
            orb_des1,
            orb_des2,
            ratio
        )

        if len(orb_good) > len(
            best_matches
        ):

            best_method = "ORB"

            best_kp1 = orb_kp1
            best_kp2 = orb_kp2
            best_matches = orb_good

    # --------------------------------------------------------
    # Match check
    # --------------------------------------------------------

    if len(best_matches) < 4:

        return {
            "success": False,
            "message": (
                "Not enough reliable matches. "
                f"{best_method} produced "
                f"{len(best_matches)} matches."
            ),
            "method": best_method,
            "keypoints1": len(best_kp1),
            "keypoints2": len(best_kp2),
            "matches": len(best_matches)
        }

    # --------------------------------------------------------
    # Homography
    # --------------------------------------------------------

    homography_result = (
        calculate_homography(
            best_kp1,
            best_kp2,
            best_matches,
            threshold
        )
    )

    if homography_result[0] is None:

        return {
            "success": False,
            "message": (
                "Feature matches were found, "
                "but RANSAC could not find a "
                "consistent transformation."
            ),
            "method": best_method,
            "keypoints1": len(best_kp1),
            "keypoints2": len(best_kp2),
            "matches": len(best_matches)
        }

    (
        H,
        mask,
        mean_error,
        median_error
    ) = homography_result

    # --------------------------------------------------------
    # Inliers
    # --------------------------------------------------------

    inliers = int(
        np.sum(mask)
    )

    matches_count = len(
        best_matches
    )

    inlier_ratio = (
        inliers /
        float(matches_count)
    )

    quality = quality_label(
        inliers,
        matches_count,
        mean_error
    )

    # --------------------------------------------------------
    # Registered image
    # --------------------------------------------------------

    h2, w2 = img2.shape[:2]

    registered = cv2.warpPerspective(
        img1,
        H,
        (w2, h2)
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        registered,
        0.5,
        img2,
        0.5,
        0
    )

    # --------------------------------------------------------
    # Inlier visualization
    # --------------------------------------------------------

    inlier_matches = [
        m
        for i, m in enumerate(
            best_matches
        )
        if mask[i]
    ]

    match_visualization = cv2.drawMatches(
        img1,
        best_kp1,
        img2,
        best_kp2,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return {
        "success": True,
        "method": best_method,
        "quality": quality,
        "keypoints1": len(best_kp1),
        "keypoints2": len(best_kp2),
        "matches": matches_count,
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "mean_error": mean_error,
        "median_error": median_error,
        "homography": H,
        "registered": registered,
        "overlay": overlay,
        "match_visualization": match_visualization
    }


# ============================================================
# LUNA AI
# ============================================================

def luna_response(question):

    q = question.lower()

    results = st.session_state.results

    if not results:

        return (
            "🌙 No registration results are available yet. "
            "Upload at least two images and run the analysis."
        )

    successful = [
        r for r in results
        if r.get("success")
    ]

    if (
        "best" in q
        or "strong" in q
    ):

        if not successful:

            return (
                "🌙 No successful registration was found."
            )

        best = max(
            successful,
            key=lambda x: x["inliers"]
        )

        return (
            f"🌙 Best result: {best['pair']}. "
            f"{best['inliers']} RANSAC inliers from "
            f"{best['matches']} matches, with "
            f"{best['mean_error']:.2f} px mean error."
        )

    if (
        "sift" in q
        or "feature" in q
    ):

        return (
            "🌙 LunarVision uses CLAHE-enhanced SIFT "
            "features with RootSIFT-style normalization. "
            "The Lowe ratio test filters ambiguous matches."
        )

    if "ransac" in q:

        return (
            "🌙 RANSAC removes geometrically inconsistent "
            "matches and estimates the homography from "
            "the remaining inliers."
        )

    if (
        "accurate" in q
        or "accuracy" in q
    ):

        return (
            "🌙 A successful feature registration does not "
            "prove absolute lunar positional accuracy. "
            "Independent control points or a validated "
            "reference dataset are required for scientific "
            "accuracy assessment."
        )

    if (
        "why" in q
        and (
            "fail" in q
            or "weak" in q
            or "reliable" in q
        )
    ):

        return (
            "🌙 Weak registration can result from limited "
            "overlap, different resolution, illumination "
            "differences, spectral differences, or too few "
            "common surface features."
        )

    return (
        f"🌙 I analysed {len(results)} pair(s). "
        f"{len(successful)} produced a homography. "
        "You can ask me about SIFT, RANSAC, the best "
        "registration, OHRC, TMC or IIRS."
    )


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown("## 🛰️ Upload Sensor Images")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("### 🔭 OHRC")

    ohrc = st.file_uploader(
        "OHRC images",
        type=[
            "jpg",
            "jpeg",
            "png",
            "tif",
            "tiff",
            "bmp"
        ],
        accept_multiple_files=True,
        key="ohrc_upload"
    )

with col2:

    st.markdown("### 🌐 TMC")

    tmc = st.file_uploader(
        "TMC images",
        type=[
            "jpg",
            "jpeg",
            "png",
            "tif",
            "tiff",
            "bmp"
        ],
        accept_multiple_files=True,
        key="tmc_upload"
    )

with col3:

    st.markdown("### 🌈 IIRS")

    iirs = st.file_uploader(
        "IIRS images",
        type=[
            "jpg",
            "jpeg",
            "png",
            "tif",
            "tiff",
            "bmp"
        ],
        accept_multiple_files=True,
        key="iirs_upload"
    )


# ============================================================
# COLLECT FILES
# ============================================================

uploaded = []

for sensor, file_list in [
    ("OHRC", ohrc),
    ("TMC", tmc),
    ("IIRS", iirs)
]:

    if file_list:

        for file in file_list:

            data = file.getvalue()

            if data:

                uploaded.append({
                    "sensor": sensor,
                    "name": file.name,
                    "bytes": data
                })


# ============================================================
# UPLOAD PREVIEWS
# ============================================================

if uploaded:

    st.markdown("## 🖼️ Uploaded Images")

    preview_columns = st.columns(
        min(4, len(uploaded))
    )

    for i, item in enumerate(
        uploaded
    ):

        with preview_columns[
            i % len(preview_columns)
        ]:

            st.markdown(
                f"**{item['sensor']}**"
            )

            preview = compact_preview(
                item["bytes"]
            )

            if preview is not None:

                st.image(
                    preview,
                    width=210
                )

            st.caption(
                item["name"]
            )


# ============================================================
# RUN REGISTRATION
# ============================================================

st.markdown("---")

if len(uploaded) >= 2:

    if st.button(
        "🚀 Run Lunar Registration",
        type="primary",
        width="stretch"
    ):

        st.session_state.results = []
        st.session_state.analysed = False

        images = []

        for item in uploaded:

            image = decode_image(
                item["bytes"]
            )

            if image is not None:

                images.append({
                    "sensor": item["sensor"],
                    "name": item["name"],
                    "image": image
                })

        if len(images) < 2:

            st.error(
                "❌ Could not decode at least two images."
            )

        else:

            pairs = list(
                itertools.combinations(
                    images,
                    2
                )
            )

            progress = st.progress(
                0,
                text="Starting registration..."
            )

            for index, (a, b) in enumerate(
                pairs
            ):

                pair = (
                    f"{a['sensor']} ↔️ "
                    f"{b['sensor']}"
                )

                progress.progress(
                    int(
                        index /
                        max(
                            len(pairs),
                            1
                        ) * 100
                    ),
                    text=(
                        f"Analysing {pair}..."
                    )
                )

                try:

                    result = register_images(
                        a["image"],
                        b["image"],
                        features,
                        lowe_ratio,
                        ransac_threshold,
                        max_size
                    )

                    result["pair"] = pair
                    result["name1"] = a["name"]
                    result["name2"] = b["name"]

                    st.session_state.results.append(
                        result
                    )

                except Exception as error:

                    st.session_state.results.append({
                        "success": False,
                        "pair": pair,
                        "name1": a["name"],
                        "name2": b["name"],
                        "message": str(error)
                    })

            progress.progress(
                100,
                text="Registration complete."
            )

            st.session_state.analysed = True

            st.success(
                "🌙 Registration analysis completed."
            )

else:

    st.info(
        "Upload at least two images to start."
    )


# ============================================================
# RESULTS
# ============================================================

if (
    st.session_state.analysed
    and st.session_state.results
):

    st.markdown("---")
    st.markdown("## 🎯 Registration Results")

    for result in st.session_state.results:

        st.markdown(
            f"### {result['pair']}"
        )

        # ----------------------------------------------------
        # FAILED
        # ----------------------------------------------------

        if not result.get("success"):

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="bad">
                        ❌ Registration not reliable
                    </div>
                    <div class="muted">
                        {result.get(
                            "message",
                            "Registration failed."
                        )}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if "keypoints1" in result:

                x1, x2, x3 = st.columns(3)

                with x1:

                    st.metric(
                        "Features A",
                        result.get(
                            "keypoints1",
                            0
                        )
                    )

                with x2:

                    st.metric(
                        "Features B",
                        result.get(
                            "keypoints2",
                            0
                        )
                    )

                with x3:

                    st.metric(
                        "Matches",
                        result.get(
                            "matches",
                            0
                        )
                    )

            continue

        # ----------------------------------------------------
        # QUALITY
        # ----------------------------------------------------

        quality = result["quality"]

        if quality == "Strong":

            icon = "✅"
            css = "good"

        elif quality == "Moderate":

            icon = "⚠️"
            css = "medium"

        else:

            icon = "❌"
            css = "bad"

        st.markdown(
            f"""
            <div class="result-card">
                <div class="{css}">
                    {icon} Registration: {quality}
                </div>

                <div class="muted">
                    Matching method: {result['method']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # SMALL METRICS
        # ----------------------------------------------------

        m1, m2, m3, m4 = st.columns(4)

        with m1:

            st.metric(
                "Matches",
                result["matches"]
            )

        with m2:

            st.metric(
                "RANSAC inliers",
                result["inliers"]
            )

        with m3:

            st.metric(
                "Inlier ratio",
                f"{result['inlier_ratio'] * 100:.1f}%"
            )

        with m4:

            st.metric(
                "Mean error",
                f"{result['mean_error']:.2f} px"
            )

        # ----------------------------------------------------
        # COMPACT RESULT IMAGES
        # ----------------------------------------------------

        with st.expander(
            "🔬 View registration images"
        ):

            tab1, tab2, tab3 = st.tabs([
                "Registered",
                "Overlay",
                "RANSAC Matches"
            ])

            with tab1:

                display = compact_result_image(
                    result["registered"],
                    max_width=500,
                    max_height=350
                )

                st.image(
                    display,
                    width=500
                )

            with tab2:

                display = compact_result_image(
                    result["overlay"],
                    max_width=500,
                    max_height=350
                )

                st.image(
                    display,
                    width=500
                )

            with tab3:

                display = compact_result_image(
                    result["match_visualization"],
                    max_width=650,
                    max_height=350
                )

                st.image(
                    display,
                    width=650
                )

            st.markdown(
                f"""
                **Source A:** {result['name1']}  
                **Source B:** {result['name2']}  
                **Method:** {result['method']}  
                **Features A:** {result['keypoints1']}  
                **Features B:** {result['keypoints2']}  
                **Median error:** {result['median_error']:.2f} px  
                **Mean error:** {result['mean_error']:.2f} px
                """
            )


# ============================================================
# LUNA AI
# ============================================================

st.markdown("---")
st.markdown("## 🌙 Ask Luna AI")

st.caption(
    "Ask Luna about registration, SIFT, RANSAC, "
    "OHRC, TMC, IIRS or matching quality."
)


for message in st.session_state.chat:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["text"]
        )


question = st.chat_input(
    "Ask Luna about your registration..."
)

if question:

    st.session_state.chat.append({
        "role": "user",
        "text": question
    })

    with st.chat_message("user"):

        st.write(
            question
        )

    answer = luna_response(
        question
    )

    st.session_state.chat.append({
        "role": "assistant",
        "text": answer
    })

    with st.chat_message("assistant"):

        st.write(
            answer
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🌙 LunarVision AI · Chandrayaan-2 Image Registration "
    "· SIFT + RootSIFT + ORB + RANSAC"
)