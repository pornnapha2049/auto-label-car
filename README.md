# 🚗 Car Segmentation & CVAT Auto-Annotator (YOLO + SAM)

ระบบตรวจจับและทำ Segmentation สำหรับ **รถยนต์และยานพาหนะ (Car / Vehicle Segmentation)** โดยผสมผสาน **YOLO** (ระบุตำแหน่ง Bounding Box และแยกประเภทรถ) ร่วมกับ **SAM (Segment Anything Model)** เพื่อปรับเส้นขอบมาร์กให้คมกริบระดับพิกเซล พร้อมแปลงเป็นฟอร์แมตมาตรฐาน **COCO 1.0** และอัปโหลดไปยังเซิร์ฟเวอร์ **CVAT (`http://1.10.131.222:8080/projects`)** อัตโนมัติผ่าน GUI

---

## 🌟 จุดเด่นของระบบ (Key Features)

1. **YOLO + SAM Refinement Pipeline (2 Major Classes)**:
   - แบ่งเป็น 2 กลุ่มคลาสหลัก:
     - **`car` (ID: 1):** รวมยานพาหนะ 4 ล้อขึ้นไปทุกประเภท (รถยนต์, รถกระบะ, รถตู้, รถ SUV, รถบัส, รถบรรทุก)
     - **`motorbike` (ID: 2):** รถมอเตอร์ไซค์ / รถจักรยานยนต์
   - YOLO ตรวจจับวัตถุ ➔ ส่ง Bounding Box Prompt ให้ SAM ตัด Mask รอบตัวรถได้อย่างแม่นยำ เส้นขอบคมเป๊ะ
   - บีบอัดลดจุดพิกัดส่วนเกินด้วย Douglas-Peucker algorithm เพื่อให้ไฟล์ COCO เบาและเรนเดอร์ลื่นไหล
2. **COCO 1.0 Annotation Format**:
   - สร้างไฟล์ `instances_default.json` และแพ็กเกจเป็น `.zip` ตามมาตรฐาน COCO 1.0 ที่ CVAT รองรับ 100%
3. **เชื่อมต่อกับ CVAT (`http://1.10.131.222:8080`)**:
   - รองรับทั้งการล็อกอินด้วย **Username / Password** หรือ **API Token**
   - **เลือก Project หรือสร้าง Project ใหม่** ได้จาก GUI โดยตรง (ระบบจะผูก Label `car, truck, bus, motorcycle` ให้อัตโนมัติ)
   - **เลือก Task ที่มีอยู่แล้ว หรือสร้าง Task ใหม่** ได้ทันที
   - รองรับ 2 โหมดการทำงานกับ CVAT:
     - **Local Images Mode:** ส่งรูปจากเครื่องไปสร้าง Task ใหม่พร้อมอัปโหลด Mask
     - **CVAT Task Mode:** ดึงรูปภาพจาก Task ที่อยู่บนเซิร์ฟเวอร์ CVAT มารัน YOLO+SAM แล้วส่ง Annotation กลับขึ้นไป
4. **เลือกขนาด Model ได้ตามต้องการ**:
   - **YOLO:** YOLOv8n / YOLOv8s / YOLOv8m / YOLOv8l / YOLO11n / YOLO11s
   - **SAM:** MobileSAM (~39MB ประมวลผลเร็วมากเหมาะกับ CPU), SAM 2 Tiny, SAM 2 Base, SAM ViT-Base
5. **หน้าจอ GUI ให้เลือกใช้งาน 2 รูปแบบ**:
   - **Desktop GUI (PyQt5):** สวยงาม ลื่นไหล มีปุ่มพรีวิวภาพ และแสดงผลแบบ Real-time
   - **Web Browser GUI (Flask):** เปิดใช้งานผ่านเบราว์เซอร์ได้ที่ `http://127.0.0.1:5000`

---

## 🚀 วิธีเปิดใช้งาน (How to Run)

### 1. เปิดผ่าน Desktop GUI (แนะนำสำหรับใช้งานบน Windows)
ดับเบิลคลิกไฟล์:
```bat
run_gui.bat
```
หรือรันผ่าน PowerShell / Terminal:
```bash
python main.py
```

### 2. เปิดผ่าน Web GUI (เปิดใน Web Browser)
ดับเบิลคลิกไฟล์:
```bat
run_web.bat
```
หรือรันผ่าน PowerShell:
```bash
python main.py --web
```
แล้วเปิดเบราว์เซอร์ไปที่: `http://127.0.0.1:5000`

---

## 📖 คู่มือการใช้งานหน้าจอ GUI

### ขั้นตอนที่ 1: เชื่อมต่อเซิร์ฟเวอร์ CVAT
1. ที่แถบ **CVAT Server**:
   - ตรวจสอบ URL: `http://1.10.131.222:8080`
   - ใส่ **Username** และ **Password** (หรือ API Token) ของบัญชีใน CVAT
   - กดปุ่ม **"🔗 Connect to CVAT"**

### ขั้นตอนที่ 2: จัดการ Project และ Task
- **Project Selection:** เลือก Project จาก Dropdown หรือกดปุ่ม **"+ New Project"** เพื่อสร้างโปรเจกต์ใหม่บนเซิร์ฟเวอร์
- **Task Selection:** เลือก Task ที่ต้องการ หรือกด **"+ New Task"** เพื่อเตรียมสร้าง Task ใหม่

### ขั้นตอนที่ 3: เลือก Model และปรับแต่งพารามิเตอร์
- **YOLO Model:** เลือกรุ่นและขนาดที่ต้องการ (เช่น `yolov8n.pt` หรือ `yolo11n.pt`)
- **SAM Model:** แนะนำ `mobile_sam.pt` สำหรับความเร็วสูงบน CPU หรือ `sam_b.pt` สำหรับความคมระดับสูงสุด
- **Vehicle Classes:** ติ๊กเลือกประเภทยานพาหนะที่ต้องการทำ Segment (Car, Truck, Bus, Motorcycle)
- **Confidence Slider:** ปรับค่าความมั่นใจในการตรวจจับ (ค่าเริ่มต้น 0.35)

### ขั้นตอนที่ 4: เลือกข้อมูลและเริ่มทำงาน
1. เลือกโฟลเดอร์ภาพที่ต้องการจากปุ่ม **"📁 Select Folder"**
2. กดปุ่ม **"🚀 Start Car Segmentation & Upload"**
3. ระบบจะแสดงพรีวิวภาพพร้อม Mask สีสันสดใส และอัปโหลด Annotations ขึ้นเซิร์ฟเวอร์ CVAT ให้โดยอัตโนมัติ

---

## 📁 โครงสร้างโปรเจกต์

```
C:\Articulus\Depa\
├── core\
│   ├── pipeline.py       # ระบบ YOLO + SAM Car Segmentation & COCO Export
│   └── cvat_manager.py   # จัดการเชื่อมต่อ CVAT API (Projects, Tasks, Uploads)
├── gui.py                # Desktop GUI (PyQt5)
├── web_app.py            # Web GUI Dashboard (Flask)
├── main.py               # จุดเริ่มโปรแกรม (เลือก desktop หรือ web)
├── run_gui.bat           # ไฟล์สำหรับดับเบิลคลิกรัน Desktop GUI
├── run_web.bat           # ไฟล์สำหรับดับเบิลคลิกรัน Web GUI
└── README.md
```
