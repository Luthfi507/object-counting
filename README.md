# Object Counting

Proyek ini adalah sistem deteksi dan penghitung objek berbasis YOLO menggunakan Ultralytics, ONNX, dan MLflow.
Tujuannya adalah melatih model deteksi objek, mengekspor ke format ONNX, mengemas model dengan MLflow, dan menyediakan demo inferensi baik dari webcam maupun video.

## Fitur utama

- Pelatihan model deteksi YOLO
- Ekspor model hasil pelatihan ke ONNX
- Logging metrik pelatihan menggunakan MLflow
- Wrapper MLflow `YOLOWrapper` untuk memuat model ONNX dan melakukan inferensi
- Demo live streaming webcam dengan deteksi objek
- Demo inferensi video menggunakan model dari MLflow

## Struktur proyek

- `requirements.txt` - daftar dependensi Python
- `.env.example` - contoh konfigurasi environment
- `models/modeling.py` - skrip pelatihan dan evaluasi model
- `models/wrapper.py` - wrapper MLflow untuk model YOLO ONNX
- `server/webcam.py` - streaming webcam dengan FastAPI
- `server/test_video.py` - pemutaran video dan inferensi model MLflow
- `utils/downloader.py` - pengunduh dataset Roboflow

## Prasyarat

- Python 3.11+ (direkomendasikan)
- CUDA GPU jika ingin menggunakan `torch` dan `onnxruntime-gpu`
- Webcam USB atau kamera internal untuk demo `server/webcam.py`

## Instalasi

1. Clone repositori:

```bash
git clone https://github.com/Luthfi507/object-counting.git
cd object-counting
```

2. Buat dan aktifkan virtual environment dengan conda:

```bash
conda create -n <env-name> python==3.12
conda activate <env-name>
```

3. Instal dependensi (gunakan `uv` untuk mempercepat instalansi):

Dokumentasi uv: https://docs.astral.sh/uv/
```bash
uv pip install -r requirements.txt
```

4. Salin konfigurasi environment dan isi sesuai dengan variabelnya:

```bash
cp .env.example .env
```

## Menyiapkan dataset

Jika ingin mengunduh dataset dari Roboflow, jalankan:

```bash
python utils/downloader.py
```

Pastikan `DATASET_DIRECTORY` sudah diatur dan dataset `Car-1` tersedia di dalam direktori tersebut.

## Menjalankan pelatihan

Pelatihan dikontrol oleh `models/modeling.py`.

```bash
python models/modeling.py
```

Skrip ini akan:

- memuat model YOLO (`yolo11m.pt`) atau mengunduhnya jika belum ada
- melatih model pada dataset `Car-1`
- mengekspor model hasil pelatihan ke ONNX
- mencatat metrik ke MLflow
- menyimpan model MLflow di dalam direktori `model`

## Menggunakan model MLflow

`models/wrapper.py` menyediakan wrapper `YOLOWrapper` yang memuat model ONNX dan menjalankan inferensi.
File ini juga dapat digunakan sebagai basis saat mendaftarkan model ke MLflow.

## Demo webcam live

Untuk menjalankan server streaming webcam dengan deteksi objek:

```bash
python server/webcam.py
```

Lalu buka browser di:

```
http://localhost:8000
```

Catatan:

- `server/webcam.py` memuat model ONNX dari path yang ditentukan di `.env`
- menggunakan perangkat kamera dengan indeks `0`

## Demo inferensi video

Untuk menjalankan demo inferensi dari file video lokal:

```bash
python server/test_video.py
```

Pastikan:

- variabel environment `MLFLOW_MODEL_NAME` dan `MLFLOW_MODEL_ALIAS` sudah diatur di `.env`

## Catatan teknis

- `models/modeling.py` menggunakan `ultralytics.YOLO` untuk pelatihan dan evaluasi
- `models/wrapper.py` mengemas model ONNX sebagai `mlflow.pyfunc.PythonModel`
- `server/webcam.py` menggunakan `FastAPI` dan `uvicorn` untuk menyajikan frame deteksi secara real-time
- `server/test_video.py` memuat model dari MLflow registry dan menampilkan output menggunakan `matplotlib`
- untuk menjalankan mlflow dengan docker, lihat [deployment](https://github.com/Luthfi507/deployment)

## Lisensi

Gunakan kode ini sesuai kebutuhan Anda. Tambahkan file lisensi (`LICENSE`) jika diperlukan untuk sharing publik.
