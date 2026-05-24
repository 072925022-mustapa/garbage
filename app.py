import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import os
import time
from PIL import Image
import json
from datetime import datetime

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR    = r"input/Garbage classification/Garbage classification"
BATCH_SIZE  = 32
EPOCHS      = 20
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

MODEL_CONFIGS = {
    "CNN": {
        "img_size":   (128, 128),
        "model_path": "garbage_cnn_model.keras",
        "desc": "Custom CNN ringan — cepat dilatih, cocok untuk eksperimen awal.",
        "arch": "Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dropout(0.5) → Dense(6)",
    },
    "MobileNetV2": {
        "img_size":   (224, 224),
        "model_path": "garbage_mobilenetv2_model.keras",
        "desc": "Transfer learning MobileNetV2 (pretrained ImageNet) — ringan dan efisien, cocok untuk deployment mobile.",
        "arch": "MobileNetV2 (frozen) → GlobalAvgPool → Dense(256, relu) → Dropout(0.5) → Dense(6, softmax)",
    },
    "InceptionV3": {
        "img_size":   (299, 299),
        "model_path": "garbage_inceptionv3_model.keras",
        "desc": "Transfer learning InceptionV3 (pretrained ImageNet) — arsitektur dalam dengan modul Inception, akurasi tinggi untuk klasifikasi gambar.",
        "arch": "InceptionV3 (frozen) → GlobalAvgPool → Dense(256, relu) → Dropout(0.5) → Dense(6, softmax)",
    },
    "EfficientNet": {
        "img_size":   (224, 224),
        "model_path": "garbage_efficientnet_model.keras",
        "desc": "Transfer learning EfficientNet (pretrained ImageNet) — efisien secara komputasi dengan akurasi tinggi, cocok untuk resource terbatas.",
        "arch": "EfficientNetB0 (frozen) → GlobalAvgPool → Dense(256, relu) → Dropout(0.5) → Dense(6, softmax)",
    },
}

# Input size yang direkomendasikan per variant EfficientNet
EFFICIENTNET_INPUT_SIZES = {
    "B0": 224, "B1": 240, "B2": 260, "B3": 300, "B4": 380,
}

st.set_page_config(
    page_title="Garbage Classification",
    page_icon="♻️",
    layout="wide",
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("♻️ Garbage Classification")

selected_model = st.sidebar.selectbox(
    "🤖 Pilih Model",
    list(MODEL_CONFIGS.keys()),
    help="CNN: custom ringan | MobileNetV2/InceptionV3/EfficientNet: transfer learning",
)

page = st.sidebar.radio(
    "Navigasi",
    ["🏠 Beranda", "🏋️ Training", "🔍 Prediksi", "📊 Evaluasi"],
)

# Ambil config model yang dipilih
cfg        = MODEL_CONFIGS[selected_model]
IMG_SIZE   = cfg["img_size"]
MODEL_PATH = cfg["model_path"]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model aktif:** `{selected_model}`")
st.sidebar.markdown(f"**Input size:** `{IMG_SIZE[0]}×{IMG_SIZE[1]}`")
st.sidebar.markdown(f"**File:** `{MODEL_PATH}`")

# ─── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource
def load_saved_model(model_path: str):
    """Load saved model from disk (cached per path)."""
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None


# `evaluate_model_file` removed: evaluation is handled in Training/Evaluasi pages.


def evaluate_model_now(model_path: str, model_name: str, img_size: tuple):
    """Evaluate a saved model on the test split and return metrics (no caching).

    Returns None if model file missing or failed to load.
    """
    if not os.path.exists(model_path):
        return None

    model = load_saved_model(model_path)
    if model is None:
        return None

    eval_img_size = img_size
    if model_name == "EfficientNet":
        try:
            actual_size = model.input_shape[1]
            eval_img_size = (actual_size, actual_size)
        except Exception:
            eval_img_size = img_size

    _, _, test_ds, _ = load_datasets(eval_img_size)
    test_ds_eval = normalize_ds(test_ds) if model_name == "CNN" else test_ds

    try:
        loss, acc = model.evaluate(test_ds_eval, verbose=0)

        y_true, y_prob = [], []
        for images, labels in test_ds_eval:
            y_true.extend(labels.numpy())
            y_prob.extend(model.predict(images, verbose=0))

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        y_pred = np.argmax(y_prob, axis=1)

        from sklearn.metrics import precision_score, recall_score, f1_score

        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        return {
            "accuracy": round(float(acc) * 100, 2),
            "precision": round(float(precision) * 100, 2),
            "recall": round(float(recall) * 100, 2),
            "f1_score": round(float(f1) * 100, 2),
            "loss": float(loss),
            "input_size": f"{eval_img_size[0]}×{eval_img_size[1]}",
        }
    except Exception:
        return None


@st.cache_data(ttl=3600)
def evaluate_model_cached(model_path: str, model_name: str, img_size: tuple, model_mtime: float):
    """Cached wrapper around `evaluate_model_now` keyed by model file mtime.

    `model_mtime` should be `os.path.getmtime(model_path)`; changing the file
    will change the cache key and force re-evaluation.
    """
    return evaluate_model_now(model_path, model_name, img_size)


@st.cache_data(ttl=3600)
def get_detailed_evaluation_cached(model_path: str, model_name: str, img_size: tuple, model_mtime: float):
    """Mengevaluasi model secara mendalam untuk mendapatkan metrik lengkap dan prediksi pada test set."""
    if not os.path.exists(model_path):
        return None

    model = load_saved_model(model_path)
    if model is None:
        return None

    eval_img_size = img_size
    if model_name == "EfficientNet":
        try:
            actual_size = model.input_shape[1]
            eval_img_size = (actual_size, actual_size)
        except Exception:
            eval_img_size = img_size

    _, _, test_ds, class_names = load_datasets(eval_img_size)
    test_ds_eval = normalize_ds(test_ds) if model_name == "CNN" else test_ds

    try:
        loss, acc = model.evaluate(test_ds_eval, verbose=0)

        y_true, y_prob = [], []
        for images, labels in test_ds_eval:
            y_true.extend(labels.numpy())
            y_prob.extend(model.predict(images, verbose=0))

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        return {
            "y_true": y_true,
            "y_prob": y_prob,
            "loss": float(loss),
            "accuracy": float(acc),
            "class_names": class_names,
        }
    except Exception:
        return None



def build_cnn(num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="custom_cnn")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mobilenetv2(num_classes: int) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_efficientnet(num_classes: int, variant: str = "B0") -> tf.keras.Model:
    """Transfer learning EfficientNet dengan variant: B0 | B1 | B2 | B3 | B4."""
    variant_map = {
        "B0": tf.keras.applications.EfficientNetB0,
        "B1": tf.keras.applications.EfficientNetB1,
        "B2": tf.keras.applications.EfficientNetB2,
        "B3": tf.keras.applications.EfficientNetB3,
        "B4": tf.keras.applications.EfficientNetB4,
    }
    # Input size yang direkomendasikan per variant
    input_size_map = {
        "B0": 224, "B1": 240, "B2": 260, "B3": 300, "B4": 380,
    }
    size = input_size_map.get(variant, 224)

    base_fn = variant_map.get(variant, tf.keras.applications.EfficientNetB0)
    base = base_fn(
        include_top=False,
        weights="imagenet",
        input_shape=(size, size, 3),
    )
    base.trainable = False  # Freeze base, latih hanya top layer

    inputs = tf.keras.Input(shape=(size, size, 3))
    # EfficientNet dari Keras sudah include normalisasi ImageNet secara internal
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name=f"efficientnet_{variant.lower()}_transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_inceptionv3(num_classes: int) -> tf.keras.Model:
    """Transfer learning InceptionV3 — input size wajib 299×299."""
    base = tf.keras.applications.InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=(299, 299, 3),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(299, 299, 3))
    # InceptionV3 butuh preprocessing khusus: skala ke [-1, 1]
    x = tf.keras.applications.inception_v3.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="inceptionv3_transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(model_name: str, num_classes: int, **kwargs) -> tf.keras.Model:
    if model_name == "CNN":
        return build_cnn(num_classes)
    elif model_name == "MobileNetV2":
        return build_mobilenetv2(num_classes)
    elif model_name == "EfficientNet":
        return build_efficientnet(num_classes, variant=kwargs.get("efficientnet_variant", "B0"))
    elif model_name == "InceptionV3":
        return build_inceptionv3(num_classes)
    raise ValueError(f"Unknown model: {model_name}")


def load_datasets(img_size: tuple):
    """Load train/val/test datasets dari DATA_DIR dengan split 70/15/15.

    Keras image_dataset_from_directory hanya support 2-way split, jadi kita:
    1. Ambil 85% sebagai pool training (train+test), 15% sebagai validation.
    2. Dari pool 85%, ambil ~82.4% (≈70/85) sebagai train dan ~17.6% (≈15/85) sebagai test.
    Hasil akhir: train ≈70%, val ≈15%, test ≈15%.
    """
    # Validation: 15% dari total
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.15,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=BATCH_SIZE,
    )
    # Pool train+test: 85% dari total
    traintest_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.15,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=BATCH_SIZE,
    )

    # Simpan class_names sebelum di-split (hilang setelah .take()/.skip())
    class_names = traintest_ds.class_names

    # Hitung jumlah batch di pool, lalu split ~82.4% train / ~17.6% test
    total_batches = traintest_ds.cardinality().numpy()
    if total_batches == tf.data.experimental.INFINITE_CARDINALITY or total_batches < 0:
        # Fallback: hitung manual
        total_batches = sum(1 for _ in traintest_ds)

    test_batches  = max(1, round(total_batches * 0.176))   # ≈15% dari total
    train_batches = total_batches - test_batches            # ≈70% dari total

    train_ds = traintest_ds.take(train_batches)
    test_ds  = traintest_ds.skip(train_batches)

    return train_ds, val_ds, test_ds, class_names


def apply_augmentation(train_ds):
    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    return train_ds.map(lambda x, y: (augment(x, training=True), y))


def normalize_ds(ds):
    """Normalisasi 0-255 → 0-1 (untuk CNN). ResNet50 pakai preprocess_input di dalam model."""
    rescale = layers.Rescaling(1.0 / 255)
    return ds.map(lambda x, y: (rescale(x), y))


def preprocess_uploaded_image(uploaded_file, img_size: tuple) -> np.ndarray:
    """Konversi file upload ke array (1, H, W, 3) float32 0-255.
    Normalisasi dilakukan di dalam model (ResNet50) atau di sini (CNN).
    """
    img = Image.open(uploaded_file).convert("RGB").resize(img_size)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def normalize_array_for_cnn(arr: np.ndarray) -> np.ndarray:
    return arr / 255.0


def plot_history(history: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["accuracy"],     label="Train Acc",  marker="o")
    axes[0].plot(history["val_accuracy"], label="Val Acc",    marker="o")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["loss"],     label="Train Loss", marker="o")
    axes[1].plot(history["val_loss"], label="Val Loss",   marker="o")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    return fig


def plot_sample_predictions(model, val_ds, class_names) -> plt.Figure:
    for images, labels in val_ds.take(1):
        preds       = model.predict(images, verbose=0)
        pred_labels = np.argmax(preds, axis=1)

        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            img = images[i].numpy()
            # Clip ke [0,1] untuk tampilan (ResNet50 bisa punya nilai di luar range)
            img = np.clip(img / 255.0 if img.max() > 1.0 else img, 0, 1)
            ax.imshow(img)
            true_lbl = class_names[labels[i]]
            pred_lbl = class_names[pred_labels[i]]
            color    = "green" if true_lbl == pred_lbl else "red"
            ax.set_title(f"True: {true_lbl}\nPred: {pred_lbl}", color=color, fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        return fig


# ─── Pages ────────────────────────────────────────────────────────────────────

# ── Beranda ──────────────────────────────────────────────────────────────────
if page == "🏠 Beranda":
    st.title("♻️ Garbage Classification — Deep Learning")
    st.markdown("""
    Aplikasi ini mengklasifikasikan gambar sampah ke dalam **6 kategori**:

    | Kelas | Deskripsi |
    |-------|-----------|
    | 📦 Cardboard | Kardus / karton |
    | 🍶 Glass | Kaca / botol kaca |
    | 🔩 Metal | Logam / kaleng |
    | 📄 Paper | Kertas |
    | 🧴 Plastic | Plastik |
    | 🗑️ Trash | Sampah umum |
    """)

    st.subheader("🤖 Model yang Tersedia")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### CNN (Custom)")
        st.info(MODEL_CONFIGS["CNN"]["desc"])
        st.code(MODEL_CONFIGS["CNN"]["arch"])
        m = load_saved_model(MODEL_CONFIGS["CNN"]["model_path"])
        if m:
            st.success("✅ Model tersimpan.")
        else:
            st.warning("⚠️ Belum dilatih.")

    with col2:
        st.markdown("### MobileNetV2 (Transfer Learning)")
        st.info(MODEL_CONFIGS["MobileNetV2"]["desc"])
        st.code(MODEL_CONFIGS["MobileNetV2"]["arch"])
        m = load_saved_model(MODEL_CONFIGS["MobileNetV2"]["model_path"])
        if m:
            st.success("✅ Model tersimpan.")
        else:
            st.warning("⚠️ Belum dilatih.")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### InceptionV3 (Transfer Learning)")
        st.info(MODEL_CONFIGS["InceptionV3"]["desc"])
        st.code(MODEL_CONFIGS["InceptionV3"]["arch"])
        m = load_saved_model(MODEL_CONFIGS["InceptionV3"]["model_path"])
        if m:
            st.success("✅ Model tersimpan.")
        else:
            st.warning("⚠️ Belum dilatih.")

    with col4:
        st.markdown("### EfficientNet (Transfer Learning)")
        st.info(MODEL_CONFIGS["EfficientNet"]["desc"])
        st.code(MODEL_CONFIGS["EfficientNet"]["arch"])
        m = load_saved_model(MODEL_CONFIGS["EfficientNet"]["model_path"])
        if m:
            st.success("✅ Model tersimpan.")
        else:
            st.warning("⚠️ Belum dilatih.")

    st.markdown("---")

    # ── Tabel Perbandingan Performa Model ─────────────────────────────────────
    st.subheader("📊 Perbandingan Performa Model")

    if not os.path.isdir(DATA_DIR):
        st.error(f"Dataset tidak ditemukan di: `{DATA_DIR}`")
    else:
        import pandas as pd

        json_file = "model_evaluasi.json"
        df = None

        # ── Cek apakah file JSON ada ────────────────────────────────────────────
        if os.path.exists(json_file):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                rows = eval_data.get("evaluasi_model", [])
                df = pd.DataFrame(rows).set_index("Model")
                st.info(f"📂 Data dimuat dari `{json_file}` (timestamp: {eval_data.get('timestamp', 'N/A')})")
            except Exception as e:
                st.error(f"❌ Gagal membaca `{json_file}`: {str(e)}")
                df = None
        else:
            # ── Evaluasi model dan buat DataFrame ────────────────────────────────
            rows = []
            with st.spinner("Memeriksa keberadaan file model dan mengevaluasi model yang tersimpan..."):
                for model_name, cfg in MODEL_CONFIGS.items():
                    model_path = cfg["model_path"]
                    exists = os.path.exists(model_path)
                    metrics = None
                    if exists:
                        model_mtime = os.path.getmtime(model_path)
                        metrics = evaluate_model_cached(model_path, model_name, cfg["img_size"], model_mtime)

                    if metrics:
                        rows.append({
                            "Model":          model_name,
                            "Accuracy (%)":   metrics["accuracy"],
                            "Precision (%)":  metrics["precision"],
                            "Recall (%)":     metrics["recall"],
                            "F1-Score (%)":   metrics["f1_score"],
                            "Loss":           metrics["loss"],
                            "Input Size":     metrics["input_size"],
                            "Status":         "Ready",
                        })
                    else:
                        rows.append({
                            "Model":          model_name,
                            "Accuracy (%)":   np.nan,
                            "Precision (%)":  np.nan,
                            "Recall (%)":     np.nan,
                            "F1-Score (%)":   np.nan,
                            "Loss":           np.nan,
                            "Input Size":     f"{cfg['img_size'][0]}×{cfg['img_size'][1]}",
                            "Status":         "Missing model file" if not exists else "Evaluation failed",
                        })

            df = pd.DataFrame(rows).set_index("Model")

            # ── Simpan evaluasi ke JSON ────────────────────────────────────────────
            try:
                df_dict = df.reset_index().to_dict(orient="records")
                for record in df_dict:
                    for key, val in record.items():
                        if isinstance(val, float) and np.isnan(val):
                            record[key] = None

                eval_data = {
                    "timestamp": datetime.now().isoformat(),
                    "evaluasi_model": df_dict,
                }

                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(eval_data, f, ensure_ascii=False, indent=2)

                st.success(f"✅ File `{json_file}` berhasil dibuat!")
            except Exception as e:
                st.warning(f"⚠️ Gagal membuat `{json_file}`: {str(e)}")

        # ── Tampilkan DataFrame ────────────────────────────────────────────────────
        if df is not None:
            numeric_cols = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]

            def highlight_best(col):
                if col.name not in numeric_cols:
                    return [""] * len(col)
                try:
                    max_val = pd.to_numeric(col, errors="coerce").max()
                    return ["background-color: #d4edda; font-weight: bold"
                            if v == max_val else "" for v in pd.to_numeric(col, errors="coerce")]
                except Exception:
                    return [""] * len(col)

            st.dataframe(
                df.style
                  .apply(highlight_best)
                  .format({
                      "Accuracy (%)":  "{:.2f}",
                      "Precision (%)": "{:.2f}",
                      "Recall (%)":    "{:.2f}",
                      "F1-Score (%)":  "{:.2f}",
                      "Loss":          "{:.4f}",
                  }, na_rep="-"),
                use_container_width=True,
            )
            st.caption("🟢 Nilai tertinggi per kolom ditandai hijau. Data dimuat dari evaluasi langsung atau file JSON.")

    st.markdown("---")
    st.markdown("""
    ### Cara Penggunaan
    1. Pilih model di **sidebar kiri**
    2. **Training** — Latih model dari dataset lokal
    3. **Prediksi** — Upload gambar dan dapatkan prediksi
    4. **Evaluasi** — Lihat grafik akurasi & loss
    """)


# ── Training ─────────────────────────────────────────────────────────────────
elif page == "🏋️ Training":
    st.title(f"🏋️ Training Model — {selected_model}")
    st.info(cfg["desc"])

    if not os.path.isdir(DATA_DIR):
        st.error(f"Dataset tidak ditemukan di: `{DATA_DIR}`")
        st.stop()

    col1, col2 = st.columns(2)
    epochs   = col1.slider("Jumlah Epoch", min_value=1, max_value=50, value=EPOCHS)
    patience = col2.slider("Early Stopping Patience", min_value=1, max_value=10, value=3)

    # Opsi variant khusus EfficientNet
    efficientnet_variant = "B0"

    if selected_model == "MobileNetV2":
        st.info("💡 MobileNetV2 menggunakan bobot pretrained ImageNet. Layer base di-freeze, hanya top layer yang dilatih.")
    elif selected_model == "EfficientNet":
        efficientnet_variant = st.selectbox(
            "Variant EfficientNet",
            ["B0", "B1", "B2", "B3", "B4"],
            index=0,
            help="B0: paling ringan (224px) → B4: paling akurat (380px, butuh RAM lebih besar)",
        )
        eff_size = EFFICIENTNET_INPUT_SIZES[efficientnet_variant]
        IMG_SIZE = (eff_size, eff_size)
        st.info(
            f"💡 EfficientNet**{efficientnet_variant}** — input size: **{eff_size}×{eff_size}px**, "
            "pretrained ImageNet, layer base di-freeze."
        )
    elif selected_model == "InceptionV3":
        st.info("💡 InceptionV3 menggunakan bobot pretrained ImageNet dengan input size **299×299px**. Layer base di-freeze, hanya top layer yang dilatih.")

    if st.button("🚀 Mulai Training", type="primary"):
        with st.spinner("Memuat dataset..."):
            train_ds, val_ds, test_ds, class_names = load_datasets(IMG_SIZE)
            num_classes = len(class_names)
            st.info(f"Kelas ditemukan: {class_names} | Input size: {IMG_SIZE}")

            train_ds = apply_augmentation(train_ds)

            # CNN perlu normalisasi manual; model transfer learning handle preprocessing di dalam model
            if selected_model == "CNN":
                train_ds = normalize_ds(train_ds)
                val_ds   = normalize_ds(val_ds)

        st.success("Dataset berhasil dimuat.")

        model = build_model(
            selected_model, num_classes,
            efficientnet_variant=efficientnet_variant,
        )

        with st.expander("📋 Arsitektur Model"):
            summary_lines = []
            model.summary(print_fn=lambda x: summary_lines.append(x))
            st.text("\n".join(summary_lines))

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )

        progress_bar        = st.progress(0)
        status_text         = st.empty()
        metrics_placeholder = st.empty()

        history_log = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                history_log["accuracy"].append(logs.get("accuracy", 0))
                history_log["val_accuracy"].append(logs.get("val_accuracy", 0))
                history_log["loss"].append(logs.get("loss", 0))
                history_log["val_loss"].append(logs.get("val_loss", 0))

                progress_bar.progress((epoch + 1) / epochs)
                status_text.text(
                    f"Epoch {epoch+1}/{epochs} — "
                    f"acc: {logs.get('accuracy', 0):.4f} | "
                    f"val_acc: {logs.get('val_accuracy', 0):.4f} | "
                    f"loss: {logs.get('loss', 0):.4f} | "
                    f"val_loss: {logs.get('val_loss', 0):.4f}"
                )

                if len(history_log["accuracy"]) > 1:
                    fig = plot_history(history_log)
                    metrics_placeholder.pyplot(fig)
                    plt.close(fig)

        with st.spinner("Training berlangsung..."):
            train_start = time.time()
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=[early_stop, StreamlitCallback()],
                verbose=0,
            )
            train_duration = time.time() - train_start

        model.save(MODEL_PATH)
        load_saved_model.clear()

        # Setelah model disimpan, update cache evaluasi untuk model ini
        try:
            model_mtime = os.path.getmtime(MODEL_PATH)
            with st.spinner("Memperbarui cache evaluasi..."):
                _ = evaluate_model_cached(MODEL_PATH, selected_model, IMG_SIZE, model_mtime)
                _ = get_detailed_evaluation_cached(MODEL_PATH, selected_model, IMG_SIZE, model_mtime)
        except Exception:
            # Jika update cache gagal, lanjutkan tanpa menghentikan alur training
            pass

        # ── Hitung metrik pada test set dan simpan ────────────────────────────
        with st.spinner("Menghitung metrik pada test set..."):
            from sklearn.metrics import precision_score, recall_score, f1_score

            test_ds_metric = normalize_ds(test_ds) if selected_model == "CNN" else test_ds

            infer_start = time.time()
            y_true_m, y_pred_m = [], []
            for imgs, lbls in test_ds_metric:
                preds_m = model.predict(imgs, verbose=0)
                y_true_m.extend(lbls.numpy())
                y_pred_m.extend(np.argmax(preds_m, axis=1))
            infer_duration = time.time() - infer_start
            total_samples  = len(y_true_m)

            y_true_m = np.array(y_true_m)
            y_pred_m = np.array(y_pred_m)

            test_loss, test_acc = model.evaluate(test_ds_metric, verbose=0)

            # Hitung precision, recall, f1_score
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision_m = precision_score(y_true_m, y_pred_m, average="macro", zero_division=0)
            recall_m = recall_score(y_true_m, y_pred_m, average="macro", zero_division=0)
            f1_m = f1_score(y_true_m, y_pred_m, average="macro", zero_division=0)

        # ── Update file model_evaluasi.json dengan metrik terbaru ────────────────
        json_file = "model_evaluasi.json"
        try:
            # Load existing JSON atau buat baru
            if os.path.exists(json_file):
                with open(json_file, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
            else:
                eval_data = {"timestamp": None, "evaluasi_model": []}

            # Cari dan update entry untuk model yang baru dilatih
            model_found = False
            for record in eval_data.get("evaluasi_model", []):
                if record.get("Model") == selected_model:
                    record.update({
                        "Accuracy (%)": round(float(test_acc) * 100, 2),
                        "Precision (%)": round(float(precision_m) * 100, 2),
                        "Recall (%)": round(float(recall_m) * 100, 2),
                        "F1-Score (%)": round(float(f1_m) * 100, 2),
                        "Loss": float(test_loss),
                        "Input Size": f"{IMG_SIZE[0]}×{IMG_SIZE[1]}",
                        "Status": "Ready",
                    })
                    model_found = True
                    break

            # Jika model tidak ditemukan, tambahkan entry baru
            if not model_found:
                eval_data["evaluasi_model"].append({
                    "Model": selected_model,
                    "Accuracy (%)": round(float(test_acc) * 100, 2),
                    "Precision (%)": round(float(precision_m) * 100, 2),
                    "Recall (%)": round(float(recall_m) * 100, 2),
                    "F1-Score (%)": round(float(f1_m) * 100, 2),
                    "Loss": float(test_loss),
                    "Input Size": f"{IMG_SIZE[0]}×{IMG_SIZE[1]}",
                    "Status": "Ready",
                })

            # Update timestamp
            eval_data["timestamp"] = datetime.now().isoformat()

            # Simpan kembali ke JSON
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(eval_data, f, ensure_ascii=False, indent=2)

            st.info(f"✅ File `{json_file}` berhasil diperbarui dengan metrik model terbaru!")
        except Exception as e:
            st.warning(f"⚠️ Gagal memperbarui `{json_file}`: {str(e)}")

        st.success(f"✅ Training selesai! Model disimpan ke `{MODEL_PATH}`.")
        st.balloons()

        fig = plot_history(history_log)
        st.pyplot(fig)
        plt.close(fig)


# ── Prediksi ─────────────────────────────────────────────────────────────────
elif page == "🔍 Prediksi":
    st.title(f"🔍 Prediksi Gambar — {selected_model}")

    model = load_saved_model(MODEL_PATH)
    if model is None:
        st.error(f"Model **{selected_model}** belum tersedia. Lakukan Training terlebih dahulu.")
        st.stop()

    uploaded_files = st.file_uploader(
        "Upload gambar sampah (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    # Untuk EfficientNet, ambil input size dari model yang tersimpan (bisa B0-B4)
    pred_img_size = IMG_SIZE
    if selected_model == "EfficientNet":
        try:
            actual_size = model.input_shape[1]  # (None, H, W, 3)
            pred_img_size = (actual_size, actual_size)
        except Exception:
            pred_img_size = IMG_SIZE

    if uploaded_files:
        cols = st.columns(min(len(uploaded_files), 3))
        for idx, uploaded_file in enumerate(uploaded_files):
            col = cols[idx % 3]
            with col:
                # Ambil array mentah (0-255), gunakan pred_img_size yang sesuai model
                img_array = preprocess_uploaded_image(uploaded_file, pred_img_size)

                # CNN butuh normalisasi 0-1; ResNet50 & ConvNeXt pakai preprocessing di dalam model
                if selected_model == "CNN":
                    input_array = normalize_array_for_cnn(img_array)
                else:
                    input_array = img_array  # ResNet50 / ConvNeXt: preprocessing ada di model

                preds      = model.predict(input_array, verbose=0)[0]
                pred_idx   = int(np.argmax(preds))
                pred_cls   = CLASS_NAMES[pred_idx]
                confidence = float(preds[pred_idx]) * 100

                st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)
                st.markdown(f"**Prediksi:** `{pred_cls}`")
                st.markdown(f"**Confidence:** `{confidence:.1f}%`")

                fig, ax = plt.subplots(figsize=(4, 2.5))
                colors  = ["#2ecc71" if i == pred_idx else "#95a5a6" for i in range(len(CLASS_NAMES))]
                ax.barh(CLASS_NAMES, preds * 100, color=colors)
                ax.set_xlabel("Probabilitas (%)")
                ax.set_xlim(0, 100)
                ax.tick_params(labelsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)


# ── Evaluasi ─────────────────────────────────────────────────────────────────
elif page == "📊 Evaluasi":
    from sklearn.metrics import (
        confusion_matrix, classification_report,
        roc_curve, auc
    )
    from sklearn.preprocessing import label_binarize

    st.title(f"📊 Evaluasi Model — {selected_model}")

    model = load_saved_model(MODEL_PATH)
    if model is None:
        st.error(f"Model **{selected_model}** belum tersedia. Lakukan Training terlebih dahulu.")
        st.stop()

    if not os.path.isdir(DATA_DIR):
        st.error(f"Dataset tidak ditemukan di: `{DATA_DIR}`")
        st.stop()

    with st.spinner("Memuat dataset test..."):
        eval_img_size = IMG_SIZE
        if selected_model == "EfficientNet":
            try:
                actual_size   = model.input_shape[1]
                eval_img_size = (actual_size, actual_size)
            except Exception:
                eval_img_size = IMG_SIZE

        _, val_ds, test_ds, class_names = load_datasets(eval_img_size)
        test_ds_eval = normalize_ds(test_ds) if selected_model == "CNN" else test_ds

    st.info("📂 Evaluasi dilakukan pada **test set (15%)** — data yang tidak pernah dilihat model saat training.")

    # ── Cek apakah data ada di model_evaluasi.json ────────────────────────────
    json_file = "model_evaluasi.json"
    metrics_from_json = None
    
    if os.path.exists(json_file):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            rows = eval_data.get("evaluasi_model", [])
            for row in rows:
                if row.get("Model") == selected_model:
                    metrics_from_json = row
                    break
            if metrics_from_json:
                st.info(f"📂 Metrik ringkas dimuat dari `{json_file}` (timestamp: {eval_data.get('timestamp', 'N/A')})")
        except Exception as e:
            st.warning(f"⚠️ Gagal membaca `{json_file}`: {str(e)}")

    # ── Load metrik dari JSON atau evaluasi model ──────────────────────────────
    # ── Ambil metrik ringkas terlebih dahulu (jika ada) agar bisa tampil cepat ─────
    acc = None
    loss = None
    precision_macro = None
    recall_macro = None
    f1_macro = None
    accuracy_macro = None

    if metrics_from_json:
        acc = metrics_from_json.get("Accuracy (%)") / 100.0 if metrics_from_json.get("Accuracy (%)") else None
        loss = metrics_from_json.get("Loss")
        precision_macro = metrics_from_json.get("Precision (%)") / 100.0 if metrics_from_json.get("Precision (%)") else None
        recall_macro = metrics_from_json.get("Recall (%)") / 100.0 if metrics_from_json.get("Recall (%)") else None
        f1_macro = metrics_from_json.get("F1-Score (%)") / 100.0 if metrics_from_json.get("F1-Score (%)") else None
        accuracy_macro = acc

    # ── Siapkan placeholder untuk metrik utama agar bisa tampil instan ───────────
    metrics_placeholder = st.empty()

    def render_metrics_cards(acc_val, loss_val, acc_macro, prec_macro, rec_macro, f1_macro_val):
        with metrics_placeholder.container():
            col1, col2 = st.columns(2)
            col1.metric("✅ Test Accuracy", f"{acc_val * 100:.2f}%" if acc_val is not None else "N/A")
            col2.metric("📉 Test Loss",     f"{loss_val:.4f}" if loss_val is not None else "N/A")

            col3, col4, col5, col6 = st.columns(4)
            col3.metric("🎯 Accuracy (macro)", f"{acc_macro * 100:.2f}%" if acc_macro is not None else "N/A")
            col4.metric("🎯 Precision (macro)", f"{prec_macro * 100:.2f}%" if prec_macro is not None else "N/A")
            col5.metric("🔁 Recall (macro)",    f"{rec_macro * 100:.2f}%" if rec_macro is not None else "N/A")
            col6.metric("⚖️ F1-Score (macro)",  f"{f1_macro_val * 100:.2f}%" if f1_macro_val is not None else "N/A")

    # Render metrik ringkas instan (bisa N/A jika tidak ada JSON)
    render_metrics_cards(acc, loss, accuracy_macro, precision_macro, recall_macro, f1_macro)

    # ── Jalankan evaluasi mendalam (menggunakan cache untuk kecepatan) ───────────
    model_mtime = os.path.getmtime(MODEL_PATH)
    with st.spinner("Mengevaluasi detail model (Confusion Matrix, ROC Curve, Classification Report)..."):
        detailed_eval = get_detailed_evaluation_cached(MODEL_PATH, selected_model, IMG_SIZE, model_mtime)

    if detailed_eval is None:
        st.error("❌ Gagal mengevaluasi detail model.")
    else:
        y_true = detailed_eval["y_true"]
        y_prob = detailed_eval["y_prob"]
        y_pred = np.argmax(y_prob, axis=1)

        # Update metrik riil dari hasil evaluasi mendalam
        acc = detailed_eval["accuracy"]
        loss = detailed_eval["loss"]
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy_macro = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall_macro    = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro        = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Render ulang metrik dengan nilai riil terakurat
        render_metrics_cards(acc, loss, accuracy_macro, precision_macro, recall_macro, f1_macro)

        # ── Classification Report ─────────────────────────────────────────────────
        st.subheader("📋 Precision, Recall, F1-Score per Kelas")
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

        report_df = {
            "Class":         class_names + ["macro avg", "weighted avg"],
            "Precision (%)": [report[c]["precision"] * 100 for c in class_names] + [report["macro avg"]["precision"] * 100, report["weighted avg"]["precision"] * 100],
            "Recall (%)":    [report[c]["recall"]    * 100 for c in class_names] + [report["macro avg"]["recall"] * 100,    report["weighted avg"]["recall"] * 100],
            "F1-Score (%)":  [report[c]["f1-score"]  * 100 for c in class_names] + [report["macro avg"]["f1-score"] * 100,  report["weighted avg"]["f1-score"] * 100],
            "Support":       [int(report[c]["support"]) for c in class_names] + [int(report["macro avg"]["support"]), int(report["weighted avg"]["support"])],
        }
        import pandas as pd
        st.dataframe(
            pd.DataFrame(report_df).set_index("Class").style.format({
                "Precision (%)": "{:.2f}", "Recall (%)": "{:.2f}", "F1-Score (%)": "{:.2f}"
            }),
            use_container_width=True,
        )

        # ── Confusion Matrix & ROC (side-by-side) ─────────────────────────────────
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))

        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.subheader("🔢 Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
            plt.colorbar(im, ax=ax_cm)
            ax_cm.set(
                xticks=range(len(class_names)), yticks=range(len(class_names)),
                xticklabels=class_names, yticklabels=class_names,
                xlabel="Predicted", ylabel="True",
            )
            plt.setp(ax_cm.get_xticklabels(), rotation=45, ha="right")
            thresh = cm.max() / 2
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax_cm.text(j, i, cm[i, j], ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with col_roc:
            st.subheader("📈 ROC Curve (One-vs-Rest)")
            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
            for i, cls in enumerate(class_names):
                try:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax_roc.plot(fpr, tpr, label=f"{cls} (AUC = {roc_auc:.2f})")
                except Exception:
                    # Jika satu kelas tidak punya sampel positif di test set, skip
                    continue
            ax_roc.plot([0, 1], [0, 1], "k--")
            ax_roc.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
            ax_roc.legend(loc="lower right")
            ax_roc.grid(True)
            plt.tight_layout()
            st.pyplot(fig_roc)
            plt.close(fig_roc)

        # ── Contoh Prediksi ───────────────────────────────────────────────────────
        st.subheader("🖼️ Contoh Prediksi pada Data Test")
        with st.spinner("Membuat prediksi..."):
            fig = plot_sample_predictions(model, test_ds_eval, class_names)
            st.pyplot(fig)
            plt.close(fig)
