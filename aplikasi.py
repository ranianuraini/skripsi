import streamlit as st
import pandas as pd
import joblib
from joblib import load
import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize


# Download NLTK data if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Function to preprocess text
def preprocess_text(text):
    # Cleaning: Remove non-alphabet characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Case folding: Convert all letters to lowercase
    text = text.lower()
    
    # Tokenizing: Split text into words
    tokens = word_tokenize(text)
    
    # Filtering: Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization: Convert words to their base form
    lemmatized_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    return ' '.join(lemmatized_tokens)

# Define sidebar navigation
st.sidebar.title("Menu:")
menu = st.sidebar.radio("", ["Dashboard Utama", "Data", "Hasil Preprocessing","Hasil Ekstraksi Fitur","Akurasi","Diagram Perbandingan", "Implementasi"])

# Dashboard Utama
if menu == "Dashboard Utama":
    st.title("Dashboard Utama")
    st.write("Selamat datang di aplikasi Analisis Sentimen Berbasis Aspek untuk Ulasan Gunung Bromo.")
    
    st.subheader("Tentang Sistem Analisis Sentimen")
    st.write("""
    Aplikasi ini menggunakan teknik analisis sentimen untuk mengolah ulasan yang diberikan oleh pengunjung Gunung Bromo.
    Ulasan tersebut dianalisis berdasarkan beberapa aspek utama, seperti:
    
    1. **Daya Tarik (Attraction)**: Mengukur seberapa menarik objek wisata Gunung Bromo bagi pengunjung.
    2. **Aksesibilitas (Accessibility)**: Menilai sejauh mana akses ke Gunung Bromo mudah dijangkau oleh pengunjung.
    3. **Citra (Image)**: Menilai bagaimana citra Gunung Bromo di mata pengunjung berdasarkan foto atau gambaran visual yang diberikan dalam ulasan.


    Sistem ini bertujuan untuk memberikan gambaran yang lebih jelas tentang persepsi pengunjung terhadap berbagai aspek yang ada di Gunung Bromo. Sentimen yang dianalisis dikategorikan sebagai positif, atau negatif yang kemudian digunakan untuk memberikan wawasan yang lebih baik bagi pengelola objek wisata dalam meningkatkan pengalaman pengunjung.

    Dengan menggunakan machine learning dan pemrosesan bahasa alami (NLP), aplikasi ini secara otomatis mengkategorikan sentimen dari setiap ulasan yang diterima berdasarkan aspek-aspek tersebut.
    """)


# Data
elif menu == "Data":
    st.title("Data")
    st.write("""
    Dataset yang digunakan merupakan kumpulan ulasan terkait Gunung Bromo, yang dikumpulkan dari berbagai sumber. 
    Data ini diproses untuk menghasilkan model berbasis aspek yang mampu memprediksi sentimen positif atau negatif.
    """)

    import pandas as pd

    try:
        # Ganti 'path_to_aspek_csv' dan 'path_to_sentimen_csv' dengan path file CSV Anda
        aspek_data_path = 'dataset/data_aspek_labelled_new.csv'
        sentimen_data_path = 'dataset/data_sentimen_labelled_new.csv'

        # Membaca data aspek dan sentimen
        aspek_df = pd.read_csv(aspek_data_path)
        sentimen_df = pd.read_csv(sentimen_data_path)

        # Menampilkan tabel data aspek
        st.write("### Data Aspek")
        st.dataframe(aspek_df)  # Menampilkan tabel data aspek

        # Menampilkan tabel data sentimen
        st.write("### Data Sentimen")
        st.dataframe(sentimen_df)  # Menampilkan tabel data sentimen

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat file CSV: {e}")


# Hasil Preprocessing
elif menu == "Hasil Preprocessing":
    st.title("Hasil Preprocessing")
    st.write("Berikut adalah langkah-langkah preprocessing yang dilakukan pada teks:")
    st.markdown("""
    - **Cleaning**: Menghapus karakter non-alfabet.
    - **Case Folding**: Mengubah semua teks menjadi huruf kecil.
    - **Tokenizing**: Memecah teks menjadi kata-kata.
    - **Filtering**: Menghapus stopwords (kata-kata umum yang tidak penting).
    - **Lemmatization**: Mengubah kata ke bentuk dasarnya.
    - **....**: .....
    """)

    # Path ke file hasil preprocessing
    preprocessed_path = "model/hasil_prepro.joblib"
    
    try:
        # Memuat hasil preprocessing dari file joblib
        df_preprocessed = joblib.load(preprocessed_path)
        
        # Menampilkan DataFrame di Streamlit
        st.write("Berikut adalah hasil preprocessing teks:")
        st.dataframe(df_preprocessed)
    except FileNotFoundError:
        st.error("File hasil preprocessing tidak ditemukan. Pastikan Anda sudah melakukan langkah preprocessing dan menyimpan hasilnya.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

elif menu == "Hasil Ekstraksi Fitur":
    st.title("Hasil Ekstraksi Fitur")

    # Penjelasan tentang ekstraksi fitur
    st.write("Berikut adalah hasil ekstraksi fitur yang digunakan dalam analisis teks:")
    st.markdown("""
    - **TF-IDF**: Menghitung Term Frequency-Inverse Document Frequency, yang menunjukkan seberapa penting suatu kata dalam dokumen.
    - **Word2Vec**: Representasi kata dalam bentuk vektor berdasarkan konteks kata tersebut dalam teks.
    """)

    # Path ke file hasil ekstraksi fitur (TF-IDF dan Word2Vec)
    tfidf_path = "model/hasil_tfidf.joblib"
    word2vec_path = "model/hasil_word2vec.joblib"

    try:
        # Memuat hasil preprocessing dari file joblib
        df_tfidf = joblib.load(tfidf_path)
        df_word2vec = joblib.load(word2vec_path)
        
        # Menampilkan DataFrame di Streamlit
        st.write("Berikut adalah hasil ekstraksi fitur tf-idf")
        st.dataframe(df_tfidf)  # Menampilkan 10 baris pertama
        st.write("Berikut adalah hasil ekstraksi fitur word2vec")
        st.dataframe(df_word2vec)  # Menampilkan 10 baris pertama
        
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")


# Menampilkan Hasil Evaluasi Akurasi
elif menu == "Akurasi":
    st.title("Akurasi TF-IDF Pada Aspek")
    
    # Daftar aspek
    list_aspek = ['Daya Tarik',  'Aksesibilitas', 'Citra']

    # Menampilkan tabel hasil evaluasi
    all_evaluations = []
    
    for aspek in list_aspek:
        # Load evaluasi hasil model
        try:
            evaluasi_per_aspek = joblib.load(f'model/evaluasi-tfidf-aspek-{aspek}.joblib')
            st.subheader(f"Evaluasi untuk Aspek: {aspek}")
            
            # Convert to DataFrame for easier display
            df_evaluasi = pd.DataFrame(evaluasi_per_aspek)
            
            # Menampilkan tabel evaluasi untuk aspek tersebut
            st.write(df_evaluasi)
            
            # Menyimpan evaluasi ke dalam list untuk diagram
            all_evaluations.append(df_evaluasi)

        except FileNotFoundError:
            st.warning(f"Data evaluasi untuk aspek {aspek} tidak ditemukan.")
    
    # Gabungkan semua evaluasi dan tampilkan dalam satu tabel
    if all_evaluations:
        all_evaluations_df = pd.concat(all_evaluations, ignore_index=True)
        st.subheader("Semua Hasil Evaluasi")
        st.write(all_evaluations_df)

    else:
        st.write("Tidak ada data evaluasi yang tersedia.")


    st.title("Akurasi TF-IDF Pada Sentimen")
    
    # Daftar aspek
    list_aspek = ['Daya Tarik', 'Aksesibilitas', 'Citra']

    # Menampilkan tabel hasil evaluasi
    all_evaluations = []
    
    for aspek in list_aspek:
        # Load evaluasi hasil model
        try:
            evaluasi_per_aspek = joblib.load(f'model/evaluasi-tfidf-sentimen-{aspek}.joblib')
            st.subheader(f"Evaluasi untuk Aspek: {aspek}")
            
            # Convert to DataFrame for easier display
            df_evaluasi = pd.DataFrame(evaluasi_per_aspek)
            
            # Menampilkan tabel evaluasi untuk aspek tersebut
            st.write(df_evaluasi)
            
            # Menyimpan evaluasi ke dalam list untuk diagram
            all_evaluations.append(df_evaluasi)

        except FileNotFoundError:
            st.warning(f"Data evaluasi untuk aspek {aspek} tidak ditemukan.")
    
    # Gabungkan semua evaluasi dan tampilkan dalam satu tabel
    if all_evaluations:
        all_evaluations_df = pd.concat(all_evaluations, ignore_index=True)
        st.subheader("Semua Hasil Evaluasi")
        st.write(all_evaluations_df)

    else:
        st.write("Tidak ada data evaluasi yang tersedia.")
    
    st.title("Akurasi WORD2VEC Pada Aspek")
    
    # Daftar aspek
    list_aspek = ['Daya Tarik', 'Aksesibilitas', 'Citra']

    # Menampilkan tabel hasil evaluasi
    all_evaluations = []
    
    for aspek in list_aspek:
        # Load evaluasi hasil model
        try:
            evaluasi_per_aspek = joblib.load(f'model/evaluasi-word2vec-aspek-{aspek}.joblib')
            st.subheader(f"Evaluasi untuk Aspek: {aspek}")
            
            # Convert to DataFrame for easier display
            df_evaluasi = pd.DataFrame(evaluasi_per_aspek)
            
            # Menampilkan tabel evaluasi untuk aspek tersebut
            st.write(df_evaluasi)
            
            # Menyimpan evaluasi ke dalam list untuk diagram
            all_evaluations.append(df_evaluasi)

        except FileNotFoundError:
            st.warning(f"Data evaluasi untuk aspek {aspek} tidak ditemukan.")
    
    # Gabungkan semua evaluasi dan tampilkan dalam satu tabel
    if all_evaluations:
        all_evaluations_df = pd.concat(all_evaluations, ignore_index=True)
        st.subheader("Semua Hasil Evaluasi")
        st.write(all_evaluations_df)

    else:
        st.write("Tidak ada data evaluasi yang tersedia.")

    st.title("Akurasi WORD2VEC Pada  Sentimen")
    
    # Daftar aspek
    list_aspek = ['Daya Tarik', 'Aksesibilitas', 'Citra']

    # Menampilkan tabel hasil evaluasi
    all_evaluations = []
    
    for aspek in list_aspek:
        # Load evaluasi hasil model
        try:
            evaluasi_per_aspek = joblib.load(f'model/evaluasi-word2vec-sentimen-{aspek}.joblib')
            st.subheader(f"Evaluasi untuk Aspek: {aspek}")
            
            # Convert to DataFrame for easier display
            df_evaluasi = pd.DataFrame(evaluasi_per_aspek)
            
            # Menampilkan tabel evaluasi untuk aspek tersebut
            st.write(df_evaluasi)
            
            # Menyimpan evaluasi ke dalam list untuk diagram
            all_evaluations.append(df_evaluasi)

        except FileNotFoundError:
            st.warning(f"Data evaluasi untuk aspek {aspek} tidak ditemukan.")
    
    # Gabungkan semua evaluasi dan tampilkan dalam satu tabel
    if all_evaluations:
        all_evaluations_df = pd.concat(all_evaluations, ignore_index=True)
        st.subheader("Semua Hasil Evaluasi")
        st.write(all_evaluations_df)

    else:
        st.write("Tidak ada data evaluasi yang tersedia.")

# Jika menu adalah "Diagram Perbandingan"
elif menu == "Diagram Perbandingan":
    st.title("Diagram Perbandingan")
    st.write("PERBANDINGAN AKURASI TF-IDF")
    
    # Menampilkan gambar pertama
    st.image('diagram_tfidf_aspek.png', use_container_width=True)
    
    # Menampilkan gambar kedua
    st.image('diagram_tfidf_sentimen.png', use_container_width=True)
    st.write("PERBANDINGAN AKURASI WORD2VEC")
    # Menampilkan gambar pertama
    st.image('diagram_word2vec_aspek.png', use_container_width=True)
    
    # Menampilkan gambar kedua
    st.image('diagram_word2vec_sentimen.png', use_container_width=True)
    

# Implementasi
elif menu == "Implementasi":
    st.title("Implementasi Sistem")
    st.write("Masukkan ulasan Anda untuk dianalisis sentimennya berdasarkan aspek tertentu.")
    
    # Input for user text
    input_text = st.text_area("Masukkan ulasan:", "")

    if st.button("Analisis"):
        # Only proceed if input_text is not empty
        if input_text.strip() != "":
            # Preprocess input text
            processed_text = preprocess_text(input_text)
            
            # Load the TF-IDF vectorizer and transform the input text
            tfidf_vectorizer = load('model/tfidf-model.joblib')
            tfidf_features = tfidf_vectorizer.transform([processed_text]).toarray()
            
            # List of aspects and sentiment labels
            list_aspek = ['attraction', 'accessibility', 'image']
            aspek_pred = {}
            label_sentimen = {-1: 'Negatif', 1: 'Positif'}
            
            # Aspect-based sentiment prediction
            for aspek in list_aspek:
                svm_model = load(f"model/tfidf-svm-aspek-{aspek}.joblib")
                pred = svm_model.predict(tfidf_features)
                
                if pred[0] == 1:
                    model_sentimen = load(f'model/tfidf-svm-sentimen-{aspek}.joblib')
                    pred_sentimen = model_sentimen.predict(tfidf_features)
                    aspek_pred[aspek] = label_sentimen[pred_sentimen[0]]
            
            # Display the results
            if aspek_pred:
                st.write("Hasil Analisis Sentimen:")
                for aspek, sentiment in aspek_pred.items():
                    st.write(f"- {aspek.capitalize()}: {sentiment}")
            else:
                st.write("Tidak ada aspek relevan yang terdeteksi dalam ulasan.")
        else:
            st.warning("Harap masukkan ulasan sebelum melakukan analisis.")
