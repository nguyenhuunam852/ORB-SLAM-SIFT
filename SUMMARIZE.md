# Tóm tắt DEBUGGING.md — Thay thế ORB bằng SIFT trong ORB-SLAM3

> Tóm tắt lại toàn bộ quá trình debug/nghiên cứu được ghi trong `DEBUGGING.md` (hơn 20 session,
> ~7900 dòng), tập trung vào 4 phần theo yêu cầu: lý do bắt đầu, baseline ORB, nỗ lực thay SIFT,
> và kết quả cuối cùng (Negative Result).

---

## 1. Lý do bắt đầu thay ORB bằng SIFT

Dự án ban đầu có một pipeline VSLAM tự viết (`SlamWorker`, dùng SIFT + P3P/5-point + BA tự
triển khai), sau đó được so sánh trực tiếp với **ORB-SLAM3 thật** (vendor nguyên bản mã nguồn gốc
vào `third_party/ORB_SLAM3`, Session 12) để có một baseline đáng tin cậy, đo trên KITTI seq00.

Sau khi pipeline tự viết cho thấy khoảng cách rất lớn so với ORB-SLAM3 gốc (xem mục 2), người
dùng đưa ra yêu cầu rõ ràng cho đề tài: **thay hẳn phần trích đặc trưng ORB bên trong lõi
ORB-SLAM3 thật bằng SIFT**, chứ không chỉ là chọn detector SIFT trong pipeline tự viết (Session
14). Đây là một yêu cầu nghiên cứu có mục đích cụ thể:

- **Ưu điểm SIFT được kỳ vọng**: SIFT là đặc trưng có độ phân biệt (distinctiveness) cao hơn,
  bất biến với scale/rotation tốt hơn ORB (đặc trưng nhị phân, dựa trên FAST corner + BRIEF),
  về lý thuyết cho match chất lượng cao hơn.
- **Phục vụ nghiên cứu**: câu hỏi đặt ra là liệu một hệ SLAM real-time cấu trúc như ORB-SLAM3
  (thiết kế xoay quanh đặc trưng nhị phân ORB — ma trận Hamming distance, DBoW2 vocabulary huấn
  luyện trên ORB, v.v.) có thể "cắm" SIFT vào và vẫn hoạt động tốt, hay tốt hơn, hay không.
- Người dùng nhấn mạnh đây là một thay đổi **kiến trúc**, không phải chỉ đổi cờ cấu hình — vì
  ORB-SLAM3 dùng ORB xuyên suốt tracking, mapping và loop-closing, không tách rời được.

---

## 2. Baseline ORB-SLAM3 gốc — vì sao đạt ATE cao (tốt)

ORB-SLAM3 gốc (chưa chỉnh sửa) được build và đo trực tiếp trên KITTI seq00, cho kết quả
**ATE RMSE ≈ 6.4–10.7 m** (so với pipeline tự viết trước đó chỉ đạt ~15–27 m khi có hỗ trợ
OXTS/IMU, và >100 m nếu chỉ dùng thị giác thuần). So sánh kiến trúc trực tiếp (Session 12) chỉ
ra các cơ chế khiến ORB-SLAM3 chính xác vượt trội:

- **Constant-velocity motion model + guided search**: dự đoán pose khung hình tiếp theo bằng mô
  hình vận tốc không đổi, sau đó chỉ tìm match trong một cửa sổ nhỏ quanh vị trí dự đoán
  (`SearchByProjection`) — thay vì match "mù" toàn ảnh rồi lọc bằng ratio test như pipeline tự viết.
- **Pose optimization phi tuyến lặp**: `Optimizer::PoseOptimization()` dùng g2o (Levenberg–
  Marquardt) tối ưu 6-DOF pose qua 4 vòng x 10 vòng lặp, loại outlier dần theo ngưỡng chi-square
  — khác hẳn cách RANSAC-rồi-dừng của PnP đơn giản.
- **Local Bundle Adjustment dựa trên covisibility graph**: chọn cửa sổ BA theo đồ thị các
  keyframe cùng nhìn thấy điểm chung, và **cố định cứng (hard-anchor)** các keyframe ngoài cửa
  sổ làm mốc tham chiếu — chạy tự động sau mỗi keyframe mới, trên thread nền riêng. (Pipeline tự
  viết chỉ dùng "soft prior" trên cửa sổ trượt, từng làm scale đơn thị bị sụp đổ khi chạy liên tục.)
- **DBoW2 vocabulary (huấn luyện sẵn trên ORB)** cho loop-closure/relocalization cực nhanh nhờ
  cơ chế cây từ vựng phân cấp (hierarchical vocabulary tree) và bucket theo "word" — lọc ứng viên
  rất hiệu quả trước khi so khớp chi tiết.

→ Bản gốc ORB-SLAM3 đạt ATE tốt không chỉ nhờ đặc trưng ORB, mà nhờ **toàn bộ kiến trúc tracking
+ mapping + loop-closing được thiết kế khớp với đặc tính của ORB** (nhị phân, nhanh, có vocabulary
huấn luyện sẵn).

---

## 3. SIFT + VSLAM: Vì sao SIFT có thể thay ORB, và những gì bắt buộc phải thay đổi

### Vì sao về mặt input/output SIFT *có thể* thay ORB

- **Input**: cả ORB và SIFT đều nhận cùng một ảnh grayscale mỗi khung hình → không cần đổi luồng
  dữ liệu đầu vào.
- **Output**: cả hai đều trả về danh sách `cv::KeyPoint` (vị trí, octave/scale, hướng) kèm mô tả
  đặc trưng (descriptor) — về mặt giao diện (interface), SIFT có thể "cắm" vào đúng những chỗ
  ORB-SLAM3 kỳ vọng nhận keypoints + descriptors mà không đổi cấu trúc dữ liệu tổng thể
  (`Frame`, `KeyFrame`, `MapPoint` vẫn giữ nguyên).
- Đây chính là lý do khiến việc thay thế **khả thi về mặt kỹ thuật**, dù descriptor SIFT (float,
  128 chiều, khoảng cách L2) khác hẳn bản chất với ORB (nhị phân 256-bit, khoảng cách Hamming).

### Những thay đổi mạnh tay bắt buộc (Session 14, fork `ORB_SLAM3_SIFT`)

Vì đây là thay đổi kiến trúc chứ không phải đổi 1 tham số, dự án đã **fork riêng** toàn bộ
`third_party/ORB_SLAM3` sang `third_party/ORB_SLAM3_SIFT` (giữ bản gốc ORB nguyên vẹn để so sánh
song song), rồi thực hiện các thay đổi lớn theo từng Stage:

1. **Stage 0 — Nghiên cứu encoding octave/layer của OpenCV SIFT**: phát hiện và sửa lỗi off-by-one
   (layer 1-indexed `[1,nOctaveLayers]` chứ không phải 0-indexed) trước khi tích hợp, tránh làm
   hỏng ngầm trọng số Bundle Adjustment ở ~25 chỗ trong `Optimizer.cc`.
2. **Stage 1 — Thay `ORBextractor` bằng wrapper `cv::SIFT`**: giữ nguyên tên/chữ ký hàm để không
   phải sửa `Frame.h`/`Tracking.h`/`Tracking.cc`.
3. **Stage 2 — Viết lại toàn bộ `ORBmatcher`**: đổi `DescriptorDistance()` từ Hamming (int, popcount
   256-bit) sang squared-L2 (float, 128-dim); đổi kiểu dữ liệu ~16 hàm từ `int` sang `float`;
   loại bỏ toàn bộ cơ chế match theo DBoW2 `FeatureVector` (bucket theo node vocabulary) — SIFT/VLAD
   không có khái niệm "node" — chuyển sang brute-force matching (bao gồm cả `SearchForTriangulation`,
   một hàm quan trọng bị bỏ sót trong kế hoạch ban đầu, nếu bỏ sót thật sẽ làm hỏng ngầm việc tạo
   map point mới ở mọi keyframe).
4. **Stage 3 — Thay thế toàn bộ hệ thống Loop-Closure/Relocalization**: vì **không tồn tại
   vocabulary DBoW2 huấn luyện sẵn cho SIFT** (đã tìm kiếm xác nhận), dự án viết mới hoàn toàn
   `VladVocabulary` (kỹ thuật VLAD — Vector of Locally Aggregated Descriptors) thay cho DBoW2,
   viết lại `KeyFrameDatabase` từ cơ chế inverted-file sang brute-force scoring, và **tự huấn
   luyện codebook VLAD (k=64)** trên toàn bộ 22 sequence KITTI (~85 triệu descriptor).
5. **Stage 4 — Hiệu chỉnh lại ngưỡng matching (`TH_HIGH`/`TH_LOW`)** dựa trên đo thực tế phân phối
   khoảng cách true-match/false-match của SIFT trên KITTI (không dùng ngưỡng cũ vốn được tinh chỉnh
   cho không gian Hamming của ORB).
6. Sau đó là hàng loạt **fix lỗi cơ chế thực sự** phát hiện trong quá trình chạy dài (Session
   15–16, Part 45–57): lỗi UB trong `KeyFrameDatabase::clearMap()` (iterator stale), lỗi mảng
   scale/sigma theo layer thay vì octave, và đặc biệt là lỗi đơn vị (`GetScaleFactor()` trả về
   giá trị config cũ 1.2 thay vì giá trị thực `2^(1/nOctaveLayers)`) khiến toàn bộ cửa sổ tìm
   kiếm `SearchByProjection` bị lệch tâm — đây là fix "sạch" duy nhất cải thiện mọi chỉ số cùng
   lúc (fails, coverage, tỉ lệ match, outlier rate) mà không đánh đổi gì.

→ Tóm lại: thay ORB bằng SIFT trong ORB-SLAM3 không chỉ là đổi 1 class trích đặc trưng, mà buộc
phải **viết lại matcher, viết lại toàn bộ hệ thống loop-closure/vocabulary, hiệu chỉnh lại ngưỡng,
và sửa nhiều lỗi đơn vị/encoding ẩn** nảy sinh từ giả định "mọi thứ đều là ORB" ăn sâu trong code
gốc.

---

## 4. Negative Result: Gap khiến SIFT không thể thay thế hoàn toàn ORB

Sau rất nhiều vòng đo đạc/tinh chỉnh (Part 45–57), kết quả cuối cùng: **SIFT-fork không thể thay
thế ORB-SLAM3 gốc**, dù đã sửa được nhiều lỗi thật sự. Các bằng chứng chính:

### a) Khoảng cách bao phủ đường đi (coverage) trên toàn sequence
Trên KITTI seq00 đầy đủ, ORB gốc đạt coverage cao hơn hẳn SIFT (dù mọi số liệu trong log đều được
đo trên cùng điều kiện/cùng đoạn frame để công bằng). Ví dụ trên 1000 frame đầu:

| | Stock ORB | SIFT (đã fix scale) | SIFT + ASIFT (cấu hình tốt nhất) |
|---|---|---|---|
| fails | 152 | 53 | 44 |
| resets | 6 | 29 | 31 |
| coverage | **74.6%** | 53.1% | 69.5% (gần nhất, vẫn thấp hơn ORB) |

### b) Gốc rễ: tỉ lệ match thấp hơn ORB gấp ~4 lần
Đo trực tiếp `SearchLocalPoints()`: **ORB match được 40.3%** số điểm map trong tầm nhìn, trong khi
**SIFT chỉ match được 10.0%** (cùng frame, cùng cấu hình). Phân tích sâu hơn cho thấy nguyên nhân
chính không phải do cửa sổ tìm kiếm hay mật độ keypoint, mà do **tỉ lệ bị từ chối vì khoảng cách
descriptor** (`dist_reject`): SIFT bị từ chối 75.9% số ứng viên tồn tại trong cửa sổ, ORB chỉ 31.9%.
Ngưỡng `TH_HIGH` được xác nhận không tự nó giải quyết được — phân phối khoảng cách true-match và
false-match của SIFT **chồng lấn gần như hoàn toàn** ở vùng ngưỡng cao, khiến việc nới ngưỡng chỉ
làm lọt thêm match sai chứ không cứu được match đúng.

### c) Nguyên nhân hình ảnh: SIFT "né" mặt đường
So sánh trực quan vị trí keypoint trên khung hình khó nhất: ORB (FAST-corner) rải dày đặc trên
**mọi bề mặt**, kể cả mặt đường nhựa phẳng (bắt cả nhiễu cường độ nhỏ), còn keypoint SIFT (cực trị
DoG) gần như **chỉ tập trung ở cấu trúc tương phản cao thật sự** (biển báo, cạnh cửa sổ, góc nhà)
và **gần như vắng mặt trên mặt đường** — đúng đặc tính lái xe forward-facing của KITTI, nơi phần
lớn khung hình là đường trống. Đây không phải lỗi code mà là **đặc tính cấu trúc** của thuật toán
phát hiện cực trị DoG so với FAST-corner.

### d) 5 nỗ lực "cứu" mật độ keypoint mặt đường — đều thất bại theo cùng một kiểu
Hạ ngưỡng contrast (0.04→0.02→0.01→0.005), kết hợp với thuật toán phân bố không gian
(`DistributeOctTree`), tiền xử lý CLAHE tăng tương phản cục bộ — **cả 5 cách đều làm coverage tệ
hơn**, theo xu hướng đơn điệu, không có điểm đảo chiều nào được tìm thấy dù thử ở mức cực đoan.
Kết luận: nhiều keypoint hơn (từ ngưỡng thấp hơn) không chuyển thành tracking tốt hơn — điểm mới
thêm vào chỉ là nhiễu, không phải cấu trúc thật.

### e) Nỗ lực cuối: ASIFT (Affine-SIFT) — thắng cục bộ, nhưng vẫn là Negative Result ở quy mô đầy đủ
Thử dùng ASIFT (`cv::AffineFeature`, mô phỏng nhiều góc nghiêng affine trước khi chạy SIFT) để
tìm lại cấu trúc mặt đường — tìm được nhiều hơn ~14.6 lần keypoint, và khi kết hợp đúng cấu hình
(`maxTilt=2` + ngưỡng chấp nhận chặt `need>=15`), **thắng SIFT thường trên đoạn ngắn** (89.2%
coverage trên 300 frame đầu, 69.5% trên 1000 frame — thu hẹp 3/4 khoảng cách với ORB). Nhưng khi
chạy **toàn bộ sequence (0–4541 frame)**, coverage sụp xuống chỉ còn **25.9%** — thấp hơn hẳn mọi
đoạn ngắn đã test — do phải reset bản đồ liên tục (121 lần) để tránh trôi (drift) tích lũy vượt
ngưỡng RMSE cho phép.

**Nguyên nhân gốc rễ (xác nhận qua chính bài báo gốc ASIFT, Morel & Yu, IPOL 2011)**: ASIFT được
thiết kế và kiểm chứng cho **wide-baseline matching** (hai ảnh cách nhau góc nhìn lớn — stereo
baseline lớn, image retrieval, ghép panorama), **không phải** cho trích đặc trưng liên tục giữa
các khung hình video liên tiếp, nơi baseline vốn rất nhỏ. Cơ chế mô phỏng nhiều góc nghiêng affine
của ASIFT tồn tại để bù cho biến dạng affine *lớn* — với cặp khung hình gần kề (baseline nhỏ), phần
lớn keypoint mô phỏng thêm **không có tương ứng thật**, nên góp phần làm tăng tỉ lệ outlier
(11.5–15% so với ~6.3% của SIFT thường) chứ không phải cấu trúc thật. Ngưỡng chấp nhận chặt chỉ
đang "lọc bớt" nhiễu này trên đoạn ngắn/dễ — không giải quyết được vấn đề gốc trên đoạn dài/khó.

### Kết luận cuối cùng

> **Negative Result**: khoảng cách giữa SIFT và ORB trong ORB-SLAM3 trên dữ liệu lái xe KITTI là
> **cấu trúc (structural)**, không phải do tinh chỉnh (tunable). SIFT cho chất lượng match/điểm
> tốt hơn trên từng điểm riêng lẻ (outlier rate 6.6% so với 17.9% của ORB — xác nhận trực tiếp),
> nhưng vì đặc tính DoG-extrema né tránh các bề mặt đồng nhất như mặt đường, SIFT không tạo đủ
> **số lượng** điểm đặc trưng ở đúng nơi ORB-SLAM3 cần để duy trì tracking liên tục trên các đoạn
> đường trống — một điểm yếu không thể khắc phục chỉ bằng hạ ngưỡng phát hiện, đổi thuật toán lọc
> không gian, tăng tương phản ảnh, hay dùng biến thể affine-invariant (ASIFT) của chính SIFT.
> ORB (FAST-corner, chấp nhận nhiễu nhẹ để đổi lấy mật độ) phù hợp hơn về mặt cấu trúc với bài
> toán VO/SLAM đơn thị, baseline nhỏ, khung hình liên tục — trong khi SIFT phù hợp hơn với các
> bài toán baseline lớn/wide-viewpoint mà nó vốn được thiết kế cho.

---

*Nguồn: `DEBUGGING.md`, các Session 12–16 và Part 45–57 (2026-07-16 → 2026-07-20).*
