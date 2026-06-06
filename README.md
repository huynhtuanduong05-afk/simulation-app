# TMV Inbound Milk Run Simulator

Đây là app Streamlit mô phỏng và đánh giá kế hoạch Milk Run inbound cho Toyota case 1.

## 1. Tool này làm gì?

Tool đi theo logic:

```text
Toyota official case data
→ Clean & aggregate daily demand
→ Build optimized Milk Run plan
→ Run distribution-based Monte Carlo simulation
→ Show decision dashboard
→ Export route plan / stop schedule / risk result
```

Điểm quan trọng: dữ liệu Toyota được dùng làm **baseline input thật**. Dữ liệu giả lập chỉ nằm ở lớp simulation, ví dụ traffic delay, loading delay, dock congestion, fuel cost change, vehicle unavailability.

## 2. Các file trong package

```text
app.py                  # Streamlit app chính
requirements.txt        # Thư viện cần cài
scenarios.json          # Scenario presets
.streamlit/config.toml  # Theme màu Toyota đỏ/trắng
README.md               # Hướng dẫn này
```

## 3. Cách chạy trên máy cá nhân

Mở terminal tại thư mục chứa các file này rồi chạy:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sau đó trình duyệt sẽ tự mở app.

## 4. Cách chạy trên Streamlit Cloud

1. Tạo một GitHub repository mới.
2. Upload toàn bộ các file trong folder này lên GitHub.
3. Vào Streamlit Cloud.
4. Chọn repository.
5. Main file path: `app.py`.
6. Deploy.

## 5. Cách dùng app

1. Upload file `dữ liệu case 1 toyota.xlsx`.
2. Nếu có distance/time matrix chính thức, upload thêm ở mục optional.
3. Chọn ngày cần mô phỏng.
4. Chọn scenario preset hoặc chỉnh các thanh trượt.
5. Bấm **Run simulation**.
6. Xem dashboard:
   - Executive Dashboard
   - Routes & Map
   - Simulation Timeline
   - What-if Comparison
   - Data & Export
7. Bấm **Download result Excel** để tải kết quả.

## 6. Distance/time matrix

Tool có thể chạy dù chưa có distance/time matrix, nhưng khi đó app sẽ dùng fallback proxy từ địa chỉ để demo logic mô phỏng. Kết quả này nên ghi rõ trong báo cáo là **assumption**.

Nếu có matrix chính thức, nên upload file Excel dạng long format:

| from | to | distance_km | duration_min |
|---|---|---:|---:|
| TMV | S01 | 25.4 | 42 |
| S01 | S02 | 3.1 | 8 |

Hoặc wide matrix với sheet chứa chữ `distance` và/hoặc `duration` trong tên sheet.

## 7. Logic simulation

App dùng distribution-based Monte Carlo simulation:

```text
travel_time = planned_time × triangular(min, mode, max)
loading_time = planned_loading_time × triangular(min, mode, max)
dock_wait = triangular(0, mode, max)
```

Kết quả chính:

- expected total cost
- P95 total cost
- truck utilization
- pickup on-time rate
- route late probability
- average delay
- CO2 estimate
- route risk ranking

## 8. Ghi chú cho báo cáo

Có thể mô tả tool như sau:

> The tool uses Toyota official case data as the baseline operational input. Synthetic data is only applied in the simulation layer through controlled probability distributions to represent traffic uncertainty, loading delay, dock congestion, demand fluctuation, and vehicle unavailability. Therefore, the model does not replace Toyota's data, but stress-tests the Milk Run plan under realistic uncertainty.
