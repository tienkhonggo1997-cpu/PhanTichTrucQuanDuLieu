# Sleep & Lifestyle Dashboard — Đồ án IE313

Đồ án môn **IE313 – Phân tích và trực quan dữ liệu** (GV Phạm Thế Sơn).
Dashboard Streamlit phân tích bộ dữ liệu *Sleep Health and Lifestyle* và dự đoán chất lượng giấc ngủ bằng hồi quy.

## Cài đặt và chạy

Cần Python 3.10 trở lên.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Phải chạy lệnh `streamlit run app.py` **từ thư mục gốc của repo**, vì app đọc file theo đường dẫn tương đối `data/Sleep_health_and_lifestyle_dataset.csv`.

Đã chạy thử với Python 3.12, streamlit 1.65, pandas 3.0, scikit-learn 1.9.

Notebook `Code_demo.ipynb` cần thêm `pip install jupyter`, và đọc file CSV ở thư mục gốc (không phải trong `data/`), nên chưa xóa bản CSV ở gốc.

## Cấu trúc

Tên file trong `views/` không trùng với tên chương, tra theo bảng này:

| Chương trên dashboard | File |
|---|---|
| 1. Tổng quan & KPIs | `views/overview.py` |
| 2. Tiền xử lý dữ liệu | `views/demographics.py` |
| 3. Trực quan hóa dữ liệu | `views/sleep_analysis.py` |
| 4. Công cụ trực quan | `views/vitals.py` |
| 5. Phân tích thăm dò | `views/explorer.py` |
| 6. Mô hình hồi quy | `views/regression.py` |
| 7. Đánh giá mô hình | `views/evaluation.py` |

- `app.py`: thanh bên, bộ lọc Giới tính / BMI, điều hướng giữa các chương.
- `modules/data_loader.py`: `load_raw_data()` trả về dữ liệu thô, `load_data()` trả về dữ liệu đã làm sạch.
- Chương 2 dùng dữ liệu thô và không chịu ảnh hưởng của bộ lọc. Các chương còn lại dùng dữ liệu đã làm sạch và đã lọc.

## Những điều cần biết về dữ liệu

Cả nhóm cần nắm 4 điểm này khi viết nhận xét và khi trả lời câu hỏi lúc báo cáo.

1. **Cột `Sleep Disorder` không thật sự thiếu dữ liệu.** File CSV ghi chữ `None` cho người không mắc rối loạn; `pd.read_csv` đọc chữ này thành NaN (219 ô). Điền lại `'None'` là khôi phục giá trị gốc.
2. **Dữ liệu trùng lặp nhiều.** 374 dòng nhưng chỉ có 132 hồ sơ khác nhau (các dòng còn lại giống hệt, chỉ khác `Person ID`). Các chương biểu đồ vẫn dùng đủ 374 dòng. Chương 7 phải chia Train/Test theo hồ sơ (xem mục dưới).
3. **Một số nghề có rất ít người:** Manager 1, Sales Representative 2, Scientist 4, Software Engineer 4. Không rút kết luận riêng cho các nghề này.
4. **Nghề nghiệp và giới tính gần như trùng nhau:** 73/73 Nurse là nữ, 69/71 Doctor là nam, 32/32 Salesperson là nam. Khác biệt "theo giới tính" và "theo nghề" vì vậy không tách được khỏi nhau.

Nguồn dữ liệu là Kaggle. Cần kiểm tra lại trang Kaggle xem tác giả có ghi đây là dữ liệu tổng hợp (synthetic) hay không, và ghi rõ trong báo cáo.

## Chương 7 đánh giá mô hình như thế nào

Bản đầu chia Train/Test ngẫu nhiên theo dòng. Với cách đó, 100/113 dòng Test có bản sao y hệt trong Train, nên Test R² không đo được khả năng dự đoán dữ liệu mới. Bản hiện tại làm như sau:

- **Chia theo hồ sơ:** các dòng giống hệt nhau luôn nằm cùng một phía (`GroupShuffleSplit`, `GroupKFold`).
- **Chọn alpha cho Ridge bằng Cross-Validation trên tập Train.** Tập Test chỉ dùng một lần để báo cáo kết quả cuối.
- **Nhận xét sinh ra từ số liệu.** Các câu như "ổn định" hay "overfitting" chỉ hiện ra khi con số thỏa điều kiện, nên vẫn đúng khi đổi bộ lọc.
- Khi bộ lọc còn dưới 10 hồ sơ, chương này hiện cảnh báo thay vì tính toán.

Kết quả với toàn bộ dữ liệu (không lọc):

| Mô hình | Train R² | Test R² | Ghi chú |
|---|---|---|---|
| Tuyến tính 4 biến | 0.889 | 0.914 | CV 5-fold: trung bình 0.875, từ 0.826 đến 0.925 |
| Ridge đa thức bậc 3 | 0.958 | 0.953 | alpha = 8.69, chọn bằng CV |
| Đa thức theo Stress Level, bậc 1 → 5 | 0.808 → 0.868 | 0.804 → 0.878 | Không thấy overfitting |

Cách nói về kết quả này khi báo cáo:

- Mô hình tuyến tính 4 biến ổn định, R² khoảng 0.88–0.91.
- Ridge bậc 3 tốt hơn ở lần chia này, nhưng **kém ổn định hơn**. Khi thử 30 cách chia khác nhau, Ridge bậc 3 thắng 23/30 lần, nhưng R² trung bình lại thấp hơn (0.855 so với 0.877) vì có lần tụt xuống âm; mô hình tuyến tính thấp nhất vẫn đạt 0.795.
- `Stress Level` chỉ có 6 giá trị (3–8), nên tăng bậc đa thức trên biến này không gây overfitting. Đây là giới hạn của dữ liệu, không phải lỗi của mô hình.
- R² ở Chương 6 được tính trên chính dữ liệu huấn luyện, nên không dùng để so sánh mô hình nào tốt hơn.

## Quy tắc viết nhận xét dưới biểu đồ

- Mỗi con số trong nhận xét phải kiểm tra lại trên dữ liệu trước khi viết. Tốt nhất là lấy từ biến trong code (f-string), không gõ tay.
- Các nhận xét viết tay ở Chương 3, 4, 5 **không đổi theo bộ lọc**; chúng chỉ đúng khi không lọc.
- Dùng "có liên hệ với" thay cho "ảnh hưởng trực tiếp" hay "là nguyên nhân". Dữ liệu này không chứng minh được quan hệ nhân quả.

## Việc còn lại

Các nhận xét sau đang sai so với dữ liệu, cần sửa:

- [ ] `views/sleep_analysis.py` (Chart 1): viết Insomnia tăng ở nhóm 51–65. Thực tế Insomnia giảm còn 7 người; chỉ Sleep Apnea tăng (33/76).
- [ ] `views/sleep_analysis.py` (Chart 3): viết Doctor ngủ "ổn định" 7–8 giờ. Thực tế Doctor biến động rộng (6.05–7.7 giờ), trung bình 6.97.
- [ ] `views/sleep_analysis.py` (Chart 3): viết Sales Representative biến thiên "rất cao". Thực tế chỉ có 2 người, cùng ngủ 5.9 giờ.
- [ ] `views/vitals.py` (4.2): nhận xét nhắc Software Engineer, nhưng biểu đồ radar vẽ Engineer.
- [ ] `views/explorer.py` (5.1): viết Accountant ngủ gần 8.0 giờ. Thực tế trung vị 7.2 giờ.
- [ ] `views/explorer.py` (5.3): viết nhóm Overweight có Insomnia "chiếm ưu thế tuyệt đối". Thực tế Insomnia 64, Sleep Apnea 65.

Việc khác:

- [ ] `Code_demo.ipynb` Chương 7 **vẫn dùng cách đánh giá cũ** (chia ngẫu nhiên theo dòng, chọn alpha trên tập Test, in ra R² 0.975). Cần sửa theo `views/evaluation.py` rồi chạy lại, nếu không số trong notebook sẽ khác số trên dashboard.
- [ ] Notebook có biểu đồ "ANOVA Visual" nhưng chưa chạy kiểm định. Thêm `scipy.stats.f_oneway` (kết quả: F = 31.9, p ≈ 1.6e-13).
- [ ] Streamlit cảnh báo `use_container_width` sắp bị bỏ. App vẫn chạy ở bản 1.65; khi nào lỗi thì đổi thành `width='stretch'`.
- [ ] Bỏ thư mục `.idea/` khỏi git và thêm vào `.gitignore`.

## Quy trình làm việc chung

- Mỗi người sửa nhóm file của mình trên một nhánh riêng, không sửa trực tiếp trên `main`.
- Trước khi đẩy lên: chạy app, mở lần lượt cả 7 chương, thử đổi bộ lọc ở thanh bên.
- Sửa nhận xét nào thì đối chiếu lại với con số trên biểu đồ ngay phía trên nó.
