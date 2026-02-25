# Dokumentasi Multimedia System - Project DETECTIFY

Dokumentasi ini disusun untuk memberikan gambaran menyeluruh mengenai sistem multimedia DETECTIFY, sebuah platform deteksi konten AI dan humanisasi teks.

## 1. Idea

### 1.1 Deskripsi
**DETECTIFY** adalah platform berbasis web yang dirancang untuk memverifikasi keaslian konten digital. Dengan kemajuan pesat AI generatif, DETECTIFY hadir untuk memberikan jawaban pasti apakah sebuah teks, gambar, video, atau audio merupakan hasil karya manusia atau buatan AI. Selain deteksi, platform ini juga menawarkan layanan "Humanizer" untuk menyesuaikan teks agar memiliki nuansa yang lebih manusiawi.

### 1.2 Latar Belakang
Didirikan pada tahun 2025, DETECTIFY lahir dari kebutuhan akan transparansi di era informasi digital yang dipenuhi konten sintetis. Kami percaya bahwa kepercayaan digital tidak seharusnya hilang karena AI, melainkan diperkuat melalui alat yang mampu membantu pengguna membedakan antara yang asli dan yang buatan, guna mendukung pengambilan keputusan yang lebih tepat.

---

## 2. Analisis

### 2.1 Market
Target pasar DETECTIFY meliputi:
- **Akademisi & Peneliti**: Memastikan keaslian karya tulis dan riset.
- **Jurnalis & Media**: Memverifikasi aset multimedia sebelum dipublikasikan.
- **Perusahaan Keamanan**: Menghindari penipuan berbasis *deepfake*.
- **Konten Kreator**: Menjaga integritas konten organik mereka.

### 2.2 Needs
- **Kecepatan**: Deteksi yang instan karena volume konten digital yang masif.
- **Akurasi**: Hasil berbasis skor kepercayaan (*confidence score*) yang transparan.
- **Multi-Format**: Satu platform untuk berbagai jenis media (Teks, Gambar, Video, Audio).
- **User Friendly**: Antarmuka yang mudah digunakan tanpa memerlukan keahlian teknis khusus.

### 2.3 Content
Konten yang ditangani oleh sistem:
- **Teks**: Deteksi AI (Luar & Dalam Negeri/Indonesia) dan Humanisasi.
- **Gambar**: Deteksi manipulasi dan AI-generated images.
- **Video**: Deteksi *deepfake* wajah dan gerakan.
- **Audio**: Deteksi kloning suara dan sintetis.

### 2.4 Medium
Platform ini menggunakan **Web Application** sebagai medium utama, memungkinkan aksesibilitas lintas perangkat (Desktop & Mobile) melalui browser.

### 2.5 Technology
- **Backend**: Flask (Python)
- **Frontend**: HTML5, Vanilla CSS (Modern design), JavaScript.
- **Database**: SQLAlchemy (SQLite/PostgreSQL).
- **Asynchronicity**: Celery & Redis untuk pemrosesan file berat.
- **Machine Learning**: PyTorch, Doc2Vec, BiLSTM (untuk deteksi teks), dan model khusus untuk visi & audio.
- **DevOps**: Docker untuk orkestrasi kontainer.

### 2.6 Cost Estimation
Estimasi biaya operasional bulanan:
- **Cloud Hosting (GPU-enabled)**: $150 - $300 (tergantung trafik).
- **Storage (S3/Object Storage)**: $20 - $50.
- **API Maintenance & Monitoring**: $30.
- **Total**: ~$200 - $380/bulan.

---

## 3. Pretesting

### 3.1 Project Goal
Membangun platform deteksi AI yang memiliki akurasi di atas 90% untuk deteksi teks dan mampu menangani unggahan multimedia secara asinkron tanpa menghambat UX pengguna.

### 3.2 Skillset Needed
- **AI/ML Engineers**: Spesialis NLP dan Computer Vision.
- **Fullstack Developers**: Mahir dalam Flask dan modern UI/UX.
- **Data Scientists**: Untuk kurasi dataset pelatihan (asli vs AI).
- **System Architects**: Merancang sistem pemrosesan latar belakang (Celery).

### 3.3 Content Outline
1. Landing Page (Introduction & Hero).
2. Dashboard Aplikasi.
3. Modul Deteksi (Text, Image, Video, Audio).
4. Modul Humanizer.
5. History & Pricing.

### 3.4 Sales & Marketing Position
DETECTIFY diposisikan sebagai "Premium Verifier". Kami tidak hanya sekadar memberikan label "AI atau Manusia", tetapi memberikan laporan transparansi yang dapat dipertanggungjawabkan (Data-driven).

### 3.5 On-paper Prototype (Optional)
*[Deskripsi: Sketsa kasar menunjukkan tata letak sidebar untuk navigasi antar fitur deteksi, dengan area drop-zone di tengah untuk unggahan file.]*

---

## 4. Prototype Development

### 4.1 UI Design - Wireframe
Konsep wireframe DETECTIFY (Fisik/Digital Tool):
- **Header**: Logo, Navigasi (Home, Apps, Model Dropdown), Theme Toggle, Get Started Button.
- **Hero Section**: Judul besar "Looking for a Real Answer?" dengan aksi "Try it for Free".
- **Features Grid**: Tiga kartu utama untuk Text, Image, dan Video detection dengan ilustrasi pendukung.
- **Footer**: Branding dan tautan cepat.

### 4.2 Style Guide
- **Typography**: Inter (Modern, Clean, Readability tinggi).
- **Color Palette**:
  - Primary: Indigo (#4F46E5)
  - Secondary: Blue (#3B82F6)
  - Background: Light Mode (Lighter Gray/White) & Dark Mode (Deep Navy/Black).
- **Components**: Rounded corners (12px-16px), Glassmorphism effects pada card, Smooth transitions (0.3s).

### 4.3 Screen Mock Up
**Main Analysis Dashboard**:
Menampilkan area unggahan tengah dengan teks "Upload your file here", tombol "Detect Now", dan sidebar sebelah kiri yang berisi status model serta riwayat singkat.

### 4.4 Content Maps
`Home -> Get Started -> Login/Register -> Dashboard -> Select Model (Text/Image/Video/Audio) -> Upload -> Result Display -> History.`

### 4.5 Story/Message
"Bringing Transparency to the AI Era" – Fokus pada penguatan kepercayaan pengguna melalui alat verifikasi yang cepat dan akurat.

### 4.6 Prototype Test Result
- UX Flow: 9/10 (Navigasi intuitif).
- Responsivitas: 8.5/10 (Berfungsi baik di mobile browser).
- Load Time: < 2 detik untuk landing page.

---

## 5. Alpha Development

### 5.1 Storyboard/Flowchart
`User Uploads File -> API Gateway receives request -> Celery Worker starts processing -> ML Model Inferences -> Database saves result -> Frontend polls/updates with Websocket or Refresh -> User views report.`

### 5.2 Asset List + Source
- **Logo**: Custom SVG Design.
- **Hero Image**: `hero-bg.jpg` (Optimized).
- **Feature Icons**: Custom illustrated jpeg/png (e.g., `text_detect.jpeg`).
- **Fonts**: Google Fonts (Inter).

### 5.3 Alpha Test Result
Fungsionalitas inti (Login, Upload, Deteksi Dasar) berhasil diimplementasikan. Bug ditemukan pada pemrosesan audio yang terkadang terhenti (delay Redis).

---

## 6. Pengembangan Tahap Beta

Memasuki periode Beta, prioritas DETECTIFY bergeser dari sekadar validasi fitur menjadi pemantapan ketahanan sistem (*system resilience*). Fokus difokuskan pada aspek *observability*, kemudahan pelacakan kinerja, serta kesiapan platform dalam menangani beban kerja nyata yang lebih dinamis.

### 6.1 Hasil Uji Coba Beta
Bagian ini merangkum pengujian komprehensif yang dilakukan untuk menyimulasikan penggunaan di luar skenario standar (*edge cases*). Pada sistem DETECTIFY, pengujian ini menitikberatkan pada pembagian hak akses pengguna, integrasi alat pantau, serta otomatisasi alur kerja (CI/CD).

#### **A. Validasi Hierarki Akun & Batasan Kuota**
- **Skenario**: Memastikan pemisahan hak akses antara pengguna *Free* (dengan limit harian) dan pengguna *Premium* (bebas akses) berjalan sesuai logika bisnis.
- **Hasil**: "Sistem secara otomatis membatasi akses dan mengirimkan notifikasi kuota habis ketika akun gratis mencoba melakukan analisis lebih dari 5 kali per hari. Di sisi lain, pengguna *Premium* dapat memproses dokumen panjang dan data multimedia berat secara simultan tanpa hambatan."

#### **B. Pemantauan Kesehatan Sistem (Grafana & Prometheus)**
- **Skenario**: Memverifikasi bahwa seluruh metrik vital (seperti penggunaan CPU, RAM, dan kecepatan respons API) terekam secara akurat dalam dasbor monitoring.
- **Hasil**: "Terintegrasi sukses dengan Grafana Alloy (ID 9666), memungkinkan tim teknis memantau beban server secara langsung. Hal ini memudahkan identifikasi lonjakan beban saat *Celery Worker* sedang memproses model deteksi yang intensif."

#### **C. Otomatisasi Alur CI/CD (GitHub Actions)**
- **Skenario**: Menguji integritas kode setiap kali ada perubahan yang masuk ke repositori utama.
- **Hasil**: "Seluruh tahapan *pipeline* di GitHub Actions menunjukkan status 'Success'. Pengetesan unit secara otomatis memastikan setiap penambahan kode baru tidak mengganggu kestabilan fitur yang sudah ada sebelumnya."

#### **D. Integritas Validasi Masukan**
- **Skenario**: Menguji daya tahan backend terhadap data yang tidak sesuai standar.
- **Hasil**: "Aplikasi mampu menangkap upaya unggah format non-.wav (seperti .mp3 atau .pdf) dan memberikan balasan *error* yang informatif. Hal ini menjamin *worker* tetap stabil dan terhindar dari potensi *crash* akibat kesalahan pemrosesan file."

### 6.2 Laporan Perbaikan Bug (Bug Reports)
Daftar masalah teknis signifikan yang berhasil diidentifikasi dan diselesaikan selama pengujian beta:

1. **Masalah KeyError pada Prediksi Teks Indonesia**
   - **Gejala**: Kegagalan sistem saat memproses teks bahasa Indonesia yang memicu pesan `KeyError: 'prediction'` pada sesi pengujian.
   - **Akar Masalah**: Adanya *mismatch* antara antarmuka `core.py` dengan format keluaran `predict_proba` khusus pada model lokal Indonesia.
   - **Solusi**: Sinkronisasi format data pada fungsi *mocking* tes agar selaras dengan output model, memastikan alur ekstraksi data berjalan lancar.
   - **Status**: **Diselesaikan (Resolved)**.

2. **Error 500 Khusus pada Layanan Premium**
   - **Gejala**: Secara spesifik hanya pengguna *Premium* yang mendapatkan respons *Internal Server Error* saat melakukan analisis teks.
   - **Akar Masalah**: Bug logika pada kueri database yang menangani pengecualian limit (*bypass*) untuk akun berbayar di lingkungan produksi.
   - **Solusi**: Perbaikan sintaks kueri SQLAlchemy untuk memastikan validasi status akun *Premium* dapat terbaca dengan tepat oleh server Flask.
   - **Status**: **Diselesaikan (Resolved)**.

3. **Visual UI Tidak Responsif pada Tampilan Mobile**
   - **Gejala**: Elemen teks dan tombol pada bagian utama (*Hero Section*) terlihat bertumpuk atau tidak sejajar ketika dibuka melalui *smartphone*.
   - **Akar Masalah**: Konfigurasi properti Flexbox yang kaku pada file CSS, sehingga tidak mampu beradaptasi dengan perubahan resolusi layar kecil.
   - **Solusi**: Implementasi *media queries* tambahan untuk mengubah orientasi tata letak menjadi vertikal dan mengatur ulang jarak elemen agar proposional di layar ponsel.
   - **Status**: **Diselesaikan (Resolved)**.

4. **Kegagalan Koneksi DB pada Jalur CI/CD**
   - **Gejala**: Pengujian otomatis di GitHub Actions gagal karena kesalahan koneksi database (`ArgumentError`).
   - **Akar Masalah**: Perbedaan format penulisan *connection string* antara environment lokal dengan konfigurasi variabel di server GitHub.
   - **Solusi**: Penyesuaian konfigurasi URL koneksi di file `.yml` YAML agar kompatibel dengan standar SQLAlchemy yang digunakan di platform CI/CD.
   - **Status**: **Diselesaikan (Resolved)**.

---

## 7. Delivery

### 7.1 Gold Master
Project Repository: [DETECTIFY Git Repository](file:///d:/Cawu%204/AI_Deepfake_Detector_and_Humanizer/testing/DETECTIFY)
Deployment Target: Dockerized Flask Application.
Status: **Ready for Production**.
