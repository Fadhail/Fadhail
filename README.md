# 📋 **PENJELASAN LENGKAP KODE TRAINING SVM**

Membuat Hyperplane, yaitu sebuah garis atau bidang pemisah terbaik antar kelompok data.

Posisi Hyperplane ini ditentukan secara eksklusif oleh Support Vector, yaitu titik-titik data terluar yang paling dekat dengan pemisah.

Tujuan utamanya adalah untuk memaksimalkan Margin, yaitu jarak atau "jalur aman" antara Hyperplane dan Support Vector.

## **🔍 OVERVIEW KODE TRAIN.PY**

Kode ini adalah script untuk melatih model SVM menggunakan dataset GoEmotions untuk klasifikasi emosi dari teks. Mari kita breakdown step-by-step:

---

## **📚 IMPORT LIBRARIES**

```python
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
```

**Penjelasan setiap import:**
- `datasets`: Library Hugging Face untuk load dataset GoEmotions
- `pandas`: Manipulasi data dalam bentuk DataFrame
- `train_test_split`: Split data menjadi training dan testing set
- `Pipeline`: Menggabungkan preprocessing dan model dalam satu workflow
- `TfidfVectorizer`: Mengkonversi teks menjadi representasi numerik TF-IDF
- `SVC`: Support Vector Classifier (implementasi SVM)
- `classification_report`: Evaluasi performa model (precision, recall, f1-score)
- `joblib`: Menyimpan dan memuat model yang sudah dilatih

---

## **📊 FUNGSI LOAD_DATA()**

```python
def load_data():
    # Load the full dataset to access label names
    full_dataset = load_dataset("go_emotions", "simplified")
    # Correct way to access label names for Sequence(ClassLabel)
    label_names = full_dataset["train"].features["labels"].feature.names
    # Now get the train split
    dataset = full_dataset["train"]
    texts = dataset["text"]
    labels = dataset["labels"]
    # For multi-label, take the first label as primary (or use another strategy)
    label_strs = [label_names[l[0]] if l else "neutral" for l in labels]
    return pd.DataFrame({'text': texts, 'label': label_strs})
```

**Step-by-step breakdown:**

### **1. Load Dataset GoEmotions**
```python
full_dataset = load_dataset("go_emotions", "simplified")
```
- Mengunduh dataset GoEmotions dari Hugging Face
- "simplified" version berisi 27 kategori emosi (vs 28 di versi lengkap)
- Dataset ini berisi ~58,000 teks dengan label emosi

### **2. Extract Label Names**
```python
label_names = full_dataset["train"].features["labels"].feature.names
```
- Mengambil nama-nama emosi: ['admiration', 'amusement', 'anger', 'annoyance', ...]
- Ini diperlukan karena label tersimpan sebagai integer index

### **3. Extract Text dan Labels**
```python
dataset = full_dataset["train"]
texts = dataset["text"]
labels = dataset["labels"]
```
- `texts`: Array berisi teks-teks seperti "I love this song!"
- `labels`: Array berisi list integer (multi-label format)

### **4. Convert Multi-label ke Single-label**
```python
label_strs = [label_names[l[0]] if l else "neutral" for l in labels]
```
- Dataset asli multi-label (satu teks bisa punya beberapa emosi)
- Kita ambil label pertama saja untuk simplifikasi
- Jika tidak ada label, set sebagai "neutral"

### **5. Return DataFrame**
```python
return pd.DataFrame({'text': texts, 'label': label_strs})
```
- Mengembalikan DataFrame dengan kolom 'text' dan 'label'

---

## **🎯 FUNGSI MAIN() - TRAINING PIPELINE**

```python
def main():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )
```

### **1. Data Splitting**
- `test_size=0.2`: 80% untuk training, 20% untuk testing
- `random_state=42`: Untuk reproducible results
- `X_train/X_test`: Teks input
- `y_train/y_test`: Label emosi target

---

## **🔧 PIPELINE CREATION**

```python
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('svm', SVC(kernel='linear', probability=True))
])
```

**Penjelasan Pipeline:**

### **Step 1: TfidfVectorizer**
- **TF (Term Frequency)**: Seberapa sering kata muncul dalam dokumen
- **IDF (Inverse Document Frequency)**: Mengurangi bobot kata yang terlalu umum
- **max_features=5000**: Hanya ambil 5000 kata paling penting
- **Output**: Mengkonversi teks menjadi vektor angka dengan 5000 dimensi

**Contoh transformasi:**
```
"I love this song!" → [0.0, 0.0, 0.8, 0.0, 0.6, 0.0, ...] (5000 angka)
```

### **Step 2: SVC (Support Vector Classifier)**
- **kernel='linear'**: Menggunakan linear kernel (terbaik untuk teks)
- **probability=True**: Mengaktifkan confidence score
- **Fungsi**: Mencari hyperplane optimal untuk memisahkan kelas emosi

---

## **🚀 TRAINING PROCESS**

```python
print("Training SVM model...")
model.fit(X_train, y_train)
```

**Apa yang terjadi saat training:**
1. TfidfVectorizer di-fit pada X_train (belajar vocabulary)
2. X_train dikonversi menjadi TF-IDF vectors
3. SVM mencari hyperplane optimal yang memisahkan 27 kelas emosi
4. Support vectors diidentifikasi
5. Margin dimaksimalkan

---

## **📊 MODEL EVALUATION**

```python
print("Evaluating model...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

**Output classification_report mencakup:**
- **Precision**: Seberapa akurat prediksi positif
- **Recall**: Seberapa banyak kasus positif yang berhasil dideteksi
- **F1-score**: Harmonic mean dari precision dan recall
- **Support**: Jumlah sampel untuk setiap kelas

**Contoh output:**
```
              precision    recall  f1-score   support
   admiration       0.82      0.79      0.80       502
    amusement       0.85      0.88      0.86       717
        anger       0.78      0.75      0.76       385
         ...
```

---

## **💾 MODEL SAVING**

```python
joblib.dump(model, "../models/svm_goemotions.pkl")
print("Model saved to models/svm_goemotions.pkl")
```

**Yang disimpan:**
- Seluruh pipeline (TfidfVectorizer + SVM)
- Vocabulary yang dipelajari TfidfVectorizer
- Support vectors dan hyperplane SVM
- Semua parameter model

**File size**: Sekitar 50-100MB (tergantung vocabulary size)

---

## **🔍 KONSEP TEKNIS PENTING**

### **1. Kenapa Linear Kernel?**
- Data teks setelah TF-IDF sudah dalam high-dimensional space
- Linear kernel efisien dan efektif untuk dimensi tinggi
- Menghindari overfitting yang sering terjadi dengan RBF kernel

### **2. Kenapa max_features=5000?**
- Balance antara informasi dan efisiensi
- Terlalu sedikit: Loss informasi penting
- Terlalu banyak: Noise dan overfitting
- 5000 adalah sweet spot untuk dataset ini

### **3. Pipeline Benefits:**
- Preprocessing otomatis saat prediksi
- Mencegah data leakage
- Mudah deploy dan maintain
- Konsistensi antara training dan inference

---

## **⚡ PERFORMANCE EXPECTATIONS**

**Training time**: 2-5 menit (tergantung hardware)
**Memory usage**: ~2-4GB RAM
**Model accuracy**: ~75-80% untuk 27 kelas
**Inference speed**: <100ms per prediction

---

## **🛠️ CARA MENJALANKAN**

```bash
# Install dependencies
pip install datasets pandas scikit-learn joblib

# Run training
python scripts/train.py

# Output akan menampilkan:
# 1. Progress training
# 2. Classification report
# 3. Konfirmasi model tersimpan
```

Kode ini adalah implementasi klasik machine learning pipeline yang solid dan production-ready untuk klasifikasi emosi menggunakan SVM! 🚀
