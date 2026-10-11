# Phân tích và trực quan dữ liệu sức khỏe giấc ngủ và lối sống

Đồ án môn học **IE313.F31.LT.CNTT – Phân tích và trực quan dữ liệu**
Giảng viên hướng dẫn: **Phạm Thế Sơn**

## Nhóm thực hiện

| STT | Họ và tên | MSSV | Phần phụ trách |
|---|---|---|---|
| 1 | Nghĩa | | |
| 2 | Tom | | |
| 3 | Luyến | | |
| 4 | Nhung | | |

## 1. Giới thiệu

Đồ án xây dựng một dashboard tương tác bằng Streamlit để phân tích bộ dữ liệu *Sleep Health and Lifestyle*, đi theo đúng trình tự của môn học: tiền xử lý, trực quan hóa, phân tích thăm dò, xây dựng mô hình hồi quy và đánh giá mô hình.

Đồ án trả lời ba câu hỏi:

1. Những yếu tố nào về lối sống và sức khỏe có liên hệ với chất lượng giấc ngủ?
2. Các dạng rối loạn giấc ngủ phân bố thế nào theo độ tuổi, thể trạng và nghề nghiệp?
3. Có thể dự đoán chất lượng giấc ngủ từ các chỉ số lối sống với độ chính xác bao nhiêu?

## 2. Dữ liệu

Bộ dữ liệu *Sleep Health and Lifestyle Dataset* lấy từ Kaggle, gồm **374 dòng** và **13 cột**.

| Cột | Ý nghĩa | Kiểu |
|---|---|---|
| Person ID | Mã định danh | Số |
| Gender | Giới tính (189 nam, 185 nữ) | Phân loại |
| Age | Tuổi (27–59) | Số |
| Occupation | Nghề nghiệp (11 nghề) | Phân loại |
| Sleep Duration | Thời lượng ngủ mỗi ngày, giờ (5.8–8.5) | Số |
| Quality of Sleep | Chất lượng giấc ngủ tự đánh giá (4–9) | Số |
| Physical Activity Level | Số phút vận động mỗi ngày (30–90) | Số |
| Stress Level | Mức độ căng thẳng tự đánh giá (3–8) | Số |
| BMI Category | Nhóm thể trạng | Phân loại |
| Blood Pressure | Huyết áp tâm thu/tâm trương | Chuỗi |
| Heart Rate | Nhịp tim lúc nghỉ, nhịp/phút (65–86) | Số |
| Daily Steps | Số bước chân mỗi ngày (3,000–10,000) | Số |
| Sleep Disorder | Rối loạn giấc ngủ: None, Insomnia, Sleep Apnea | Phân loại |

Hai đặc điểm của dữ liệu ảnh hưởng đến cách phân tích:

- **Giá trị `None` bị đọc thành dữ liệu khuyết.** Cột `Sleep Disorder` ghi chữ `None` cho người không mắc rối loạn; `pd.read_csv` đọc chữ này thành NaN (219 ô). Đây không phải dữ liệu thiếu thật, nên nhóm điền lại bằng `'None'`.
- **Nhiều dòng trùng lặp.** Trong 374 dòng chỉ có **132 hồ sơ khác nhau**; các dòng còn lại giống hệt nhau, chỉ khác `Person ID`. Các biểu đồ vẫn dùng đủ 374 dòng, còn phần đánh giá mô hình được thiết kế riêng để xử lý vấn đề này (mục 5.7).

## 3. Công nghệ sử dụng

| Mục đích | Thư viện |
|---|---|
| Giao diện dashboard | Streamlit |
| Xử lý dữ liệu | pandas, NumPy |
| Biểu đồ tương tác | Plotly |
| Biểu đồ thống kê | Matplotlib, Seaborn |
| Mô hình hồi quy và đánh giá | scikit-learn |

## 4. Cài đặt và chạy

Yêu cầu Python 3.10 trở lên.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Chạy lệnh `streamlit run app.py` từ thư mục gốc của repo. Dashboard mở tại `http://localhost:8501`.

Đã kiểm thử với Python 3.12, Streamlit 1.65, pandas 3.0, scikit-learn 1.9.

### Cấu trúc thư mục

```
├── app.py                    # Thanh bên, bộ lọc, điều hướng giữa các chương
├── modules/
│   └── data_loader.py        # Đọc dữ liệu thô và dữ liệu đã làm sạch
├── views/
│   ├── overview.py           # Chương 1: Tổng quan & KPIs
│   ├── demographics.py       # Chương 2: Tiền xử lý dữ liệu
│   ├── sleep_analysis.py     # Chương 3: Trực quan hóa dữ liệu
│   ├── vitals.py             # Chương 4: Công cụ trực quan
│   ├── explorer.py           # Chương 5: Phân tích thăm dò
│   ├── regression.py         # Chương 6: Mô hình hồi quy
│   └── evaluation.py         # Chương 7: Đánh giá mô hình
├── data/
│   └── Sleep_health_and_lifestyle_dataset.csv
├── Code_demo.ipynb           # Notebook minh họa theo từng chương
└── requirements.txt
```

Thanh bên có bộ lọc theo **Giới tính** và **Nhóm BMI**, áp dụng cho mọi chương trừ Chương 2 (chương này luôn hiển thị dữ liệu thô đầy đủ).

## 5. Nội dung và kết quả

Các con số dưới đây tính trên toàn bộ dữ liệu, không áp dụng bộ lọc.

### 5.1. Chương 1 – Tổng quan

Trang tổng quan hiển thị năm chỉ số chính, tỷ lệ rối loạn giấc ngủ và phân bố BMI theo giới tính.

| Chỉ số | Giá trị trung bình |
|---|---|
| Thời lượng ngủ | 7.13 giờ |
| Chất lượng giấc ngủ | 7.31 / 10 |
| Mức độ căng thẳng | 5.39 / 10 |
| Số bước chân mỗi ngày | 6,817 |

Về rối loạn giấc ngủ: 58.6% không mắc, 20.9% mắc Sleep Apnea (ngưng thở khi ngủ), 20.6% mắc Insomnia (mất ngủ).

### 5.2. Chương 2 – Tiền xử lý dữ liệu

Trang này cho thấy dữ liệu trước và sau khi làm sạch, kèm mã nguồn của từng bước:

1. Điền `'None'` cho 219 ô khuyết ở cột `Sleep Disorder`.
2. Gộp nhãn `Normal Weight` vào `Normal` ở cột `BMI Category`. Sau khi gộp: Normal 216, Overweight 148, Obese 10.
3. Tách cột `Blood Pressure` thành hai cột số `Systolic_BP` và `Diastolic_BP`.
4. Chia tuổi thành ba nhóm: 20–35, 36–50, 51–65.

Notebook `Code_demo.ipynb` minh họa thêm các kỹ thuật chuẩn hóa Min-Max, Z-score, phân nhóm (binning) và mã hóa biến phân loại.

### 5.3. Chương 3 – Trực quan hóa dữ liệu

Bốn biểu đồ tương tác: diện tích tích lũy theo nhóm tuổi, phân tán huyết áp, hộp thời lượng ngủ theo nghề nghiệp, và bản đồ nhiệt giữa căng thẳng và chất lượng giấc ngủ.

**Rối loạn giấc ngủ theo nhóm tuổi**

| Nhóm tuổi | Không mắc | Insomnia | Sleep Apnea |
|---|---|---|---|
| 20–35 | 80 | 5 | 9 |
| 36–50 | 103 | 65 | 36 |
| 51–65 | 36 | 7 | 33 |

Nhóm 20–35 hầu hết không mắc rối loạn. Insomnia tập trung ở nhóm 36–50, còn Sleep Apnea chiếm tỷ lệ cao nhất ở nhóm 51–65 (33/76 người).

**Huyết áp theo rối loạn giấc ngủ**

| Nhóm | Tâm thu trung bình | Tâm trương trung bình |
|---|---|---|
| Không mắc | 124.0 | 81.0 |
| Insomnia | 132.0 | 86.9 |
| Sleep Apnea | 137.8 | 92.7 |

Nhóm Sleep Apnea có huyết áp cao nhất ở cả hai chỉ số.

**Căng thẳng và chất lượng giấc ngủ.** Chất lượng giấc ngủ trung bình giảm từ 8.97 ở mức stress 3 xuống 5.86 ở mức stress 8.

### 5.4. Chương 4 – Công cụ trực quan

Sáu dạng biểu đồ nâng cao: diện tích tích lũy, radar so sánh bốn nghề đông nhất (Nurse, Doctor, Engineer, Lawyer), ma trận tương quan, phân tán kèm phân bố biên, tọa độ song song, và ma trận phân tán theo cặp biến.

Các hệ số tương quan Pearson đáng chú ý:

| Cặp biến | Hệ số r |
|---|---|
| Thời lượng ngủ – Chất lượng giấc ngủ | 0.88 |
| Căng thẳng – Chất lượng giấc ngủ | −0.90 |
| Căng thẳng – Thời lượng ngủ | −0.81 |
| Số bước chân – Mức vận động | 0.77 |

Trong bốn nghề trên biểu đồ radar, Engineer có thời lượng ngủ (7.99 giờ) và chất lượng giấc ngủ (8.41) cao nhất, mức căng thẳng thấp nhất (3.89). Nurse có mức vận động cao nhất (78.6 phút/ngày).

### 5.5. Chương 5 – Phân tích thăm dò

Năm biểu đồ thống kê: hộp thời lượng ngủ theo nghề, phân bố nhịp tim, rối loạn giấc ngủ theo BMI, phân tán căng thẳng và chất lượng giấc ngủ, hộp số bước chân theo rối loạn giấc ngủ.

**Rối loạn giấc ngủ theo nhóm BMI**

| Nhóm BMI | Không mắc | Insomnia | Sleep Apnea |
|---|---|---|---|
| Normal | 200 | 9 | 7 |
| Overweight | 19 | 64 | 65 |
| Obese | 0 | 4 | 6 |

Đây là mối liên hệ rõ nhất trong dữ liệu: 93% người có BMI bình thường không mắc rối loạn, trong khi 87% người thừa cân và toàn bộ 10 người béo phì đều mắc một dạng rối loạn. Ở nhóm thừa cân, Insomnia và Sleep Apnea gần như ngang nhau.

**Thời lượng ngủ theo nghề nghiệp** (các nghề có từ 30 người trở lên): Engineer có trung vị cao nhất (8.3 giờ), Salesperson thấp nhất (6.4 giờ). Doctor và Nurse có độ phân tán rộng nhất, với khoảng tứ phân vị lần lượt là 6.05–7.7 và 6.1–8.1 giờ.

### 5.6. Chương 6 – Mô hình hồi quy

Biến mục tiêu là `Quality of Sleep`. Bốn mô hình được xây dựng trên toàn bộ dữ liệu:

| Mô hình | Biến đầu vào | R² | MSE |
|---|---|---|---|
| Hồi quy tuyến tính đơn | Sleep Duration | 0.780 | 0.314 |
| Hồi quy đa thức bậc 2 | Stress Level | 0.809 | 0.273 |
| Hồi quy tuyến tính đa biến | Sleep Duration, Stress Level, Heart Rate, Physical Activity Level | 0.897 | 0.147 |
| Pipeline: chuẩn hóa + đa thức bậc 2 + tuyến tính | Bốn biến trên | 0.959 | 0.059 |

Phương trình hồi quy tuyến tính đơn: `Quality of Sleep = −2.163 + 1.329 × Sleep Duration`.

R² ở chương này được tính trên chính dữ liệu dùng để huấn luyện, nên mô hình càng phức tạp thì R² càng cao. Khả năng dự đoán trên dữ liệu mới được đánh giá ở Chương 7.

### 5.7. Chương 7 – Đánh giá mô hình

**Phương pháp.** Vì dữ liệu có nhiều dòng trùng lặp, nếu chia Train/Test ngẫu nhiên theo dòng thì 100/113 dòng Test sẽ có bản sao y hệt trong Train, và R² trên tập Test không còn phản ánh khả năng dự đoán dữ liệu mới. Nhóm xử lý như sau:

- Chia Train/Test (70/30) **theo hồ sơ** bằng `GroupShuffleSplit`: các dòng giống hệt nhau luôn nằm cùng một tập. Kết quả là 274 dòng Train và 100 dòng Test.
- Cross-Validation 5-fold cũng chia theo hồ sơ bằng `GroupKFold`.
- Hệ số điều chuẩn alpha của Ridge được chọn bằng Cross-Validation trên tập Train. Tập Test chỉ dùng một lần để báo cáo kết quả cuối.

**Kết quả**

| Mô hình | Train R² | Test R² | Ghi chú |
|---|---|---|---|
| Tuyến tính 4 biến | 0.889 | 0.914 | CV 5-fold: trung bình 0.875, từ 0.826 đến 0.925 |
| Ridge trên đa thức bậc 3 | 0.958 | 0.953 | alpha = 8.69 |

**Underfitting và overfitting theo bậc đa thức** (một biến `Stress Level`)

| Bậc | Train R² | Test R² |
|---|---|---|
| 1 | 0.808 | 0.804 |
| 2 | 0.808 | 0.807 |
| 3 | 0.809 | 0.806 |
| 4 | 0.861 | 0.877 |
| 5 | 0.868 | 0.878 |

**Nhận xét**

- Mô hình tuyến tính 4 biến có Train R² và Test R² gần nhau và ổn định qua các fold, không có dấu hiệu overfitting.
- Ridge trên đa thức bậc 3 cho Test R² cao hơn (0.953 so với 0.914), nhưng kém ổn định hơn. Khi lặp lại với 30 cách chia Train/Test khác nhau, Ridge bậc 3 cho kết quả tốt hơn ở 23 lần, song R² trung bình lại thấp hơn (0.855 so với 0.877) do có lần giảm xuống dưới 0. Mô hình tuyến tính thấp nhất vẫn đạt 0.795.
- Với biến `Stress Level`, bậc 1 đến 3 cho kết quả gần như nhau (underfitting nhẹ), bậc 4 và 5 cải thiện rõ. Không quan sát thấy overfitting, vì biến này chỉ có 6 giá trị khác nhau nên đa thức bậc 5 đã là mức phức tạp tối đa.

## 6. Kết luận

1. **Căng thẳng và thời lượng ngủ là hai yếu tố liên hệ mạnh nhất với chất lượng giấc ngủ**, với hệ số tương quan lần lượt là −0.90 và 0.88.
2. **Thể trạng liên hệ rất rõ với rối loạn giấc ngủ**: 93% người có BMI bình thường không mắc rối loạn, so với 13% ở nhóm thừa cân.
3. **Rối loạn giấc ngủ thay đổi theo tuổi**: Insomnia tập trung ở nhóm 36–50, Sleep Apnea chiếm tỷ lệ cao nhất ở nhóm 51–65.
4. **Chất lượng giấc ngủ dự đoán được khá tốt**: mô hình tuyến tính với bốn biến đạt R² khoảng 0.88–0.91 trên dữ liệu chưa từng thấy. Mô hình phức tạp hơn có thể đạt R² cao hơn nhưng kém ổn định.

## 7. Hạn chế và hướng phát triển

**Hạn chế**

- Dữ liệu nhỏ và trùng lặp nhiều: chỉ có 132 hồ sơ khác nhau, nên kết quả mô hình thay đổi đáng kể theo cách chia dữ liệu.
- Một số nghề có quá ít người (Manager 1, Sales Representative 2, Scientist 4, Software Engineer 4) nên không thể rút ra kết luận cho các nghề này.
- Nghề nghiệp và giới tính gần như trùng nhau (toàn bộ 73 Nurse là nữ, 69/71 Doctor là nam), nên không tách được ảnh hưởng của hai yếu tố này.
- Dữ liệu thu thập tại một thời điểm, chỉ cho thấy mối liên hệ chứ không chứng minh được quan hệ nhân quả.
- `Quality of Sleep` và `Stress Level` là điểm tự đánh giá, chỉ nhận 6 giá trị nguyên.
- Phần nhận xét dưới các biểu đồ ở Chương 3, 4 và 5 được viết cho toàn bộ dữ liệu và không tự cập nhật khi thay đổi bộ lọc.

**Hướng phát triển**

- Bổ sung kiểm định thống kê (ANOVA, Chi-square) cho các khác biệt giữa các nhóm.
- Xây dựng mô hình phân loại để dự đoán dạng rối loạn giấc ngủ.
- Thử nghiệm trên bộ dữ liệu lớn hơn và không trùng lặp.
