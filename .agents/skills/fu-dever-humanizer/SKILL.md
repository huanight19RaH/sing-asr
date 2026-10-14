---
name: fu-dever-humanizer
description: Quét và loại bỏ triệt để các dấu hiệu văn mẫu AI (AI tells, Not-X-but-Y, staged openers, emoji spam, inflated buzzwords) khỏi văn bản tiếng Việt của FU-DEVER, biến copy thành giọng điệu chân thực, tự nhiên, đậm chất builder/developer.
---

# FU-DEVER Humanizer Skill

Kỹ năng chuyên sâu dùng để biên tập, "giải phóng" văn bản khỏi các cấu trúc máy móc của chatbot và AI, áp dụng hệ thống 26 dấu hiệu nhận diện AI (dựa trên Wikipedia AI Cleanup & humanizer v3.1.0 framework) chuẩn hóa riêng cho văn hóa sinh viên công nghệ FU-DEVER.

## 1. Các "tội đồ" văn mẫu AI cần triệt tiêu ngay lập tức

### A. Cấu trúc "Not X but Y" (Không chỉ X mà còn Y)
- **Dấu hiệu:** "Tại CLB, bạn không chỉ học lý thuyết mà còn được làm sản phẩm...", "Đây không đơn thuần là một cuộc thi, mà là một hành trình..."
- **Cách xử lý:** Bỏ vế phủ định gượng ép, nói thẳng vào giá trị thực tế: *"Tại CLB, bạn trực tiếp tham gia phát triển sản phẩm thực tế..."*

### B. Mở bài dàn cảnh sáo rỗng (Staged Run-up & Rhetorical Questions)
- **Dấu hiệu:** "Bước vào cánh cổng giảng đường đại học, bạn đang tìm kiếm...", "Bạn có bao giờ tự hỏi...", "Dưới đây là tất cả những gì bạn cần biết..."
- **Cách xử lý:** Cắt bỏ đoạn mào đầu. Bắt đầu ngay bằng một observation thực tế, một mâu thuẫn thú vị (Contrarian hook), hoặc đi thẳng vào vấn đề.

### C. Từ ngữ đao to búa lớn & Hô khẩu hiệu giáo điều (Inflated Significance)
- **Từ cấm:** "cơ hội vàng", "đừng để thanh xuân/năm tháng đại học trôi qua lặng lẽ", "bùng nổ rực rỡ", "vô cùng ý nghĩa và bổ ích", "hành trình đáng nhớ", "cháy hết mình cùng đam mê".
- **Cách xử lý:** Thay bằng chi tiết cụ thể: cày Hacknight, thức đêm debug cùng anh em, tự dựng pipeline CI/CD, bốc quà Club Day.

### D. Rải Emoji làm đạn hoa trang trí (Decorative Bullet Emojis)
- **Dấu hiệu:** Mỗi đầu dòng là một icon: 🚀, 💡, 🔥, 👉, ⚡, 🌟, 📌, ⏰, 🎪, 🎙️, ✨.
- **Quy tắc cứng:** Giới hạn tối đa **0 – 3 emoji trong toàn bài**, chỉ dùng khi có mục đích diễn đạt cảm xúc hoặc nhịp điệu tự nhiên. Tuyệt đối không dùng emoji làm bullet point thay cho dấu gạch đầu dòng chuẩn.

### E. Danh sách bôi đậm từng chữ theo công thức (Bold-as-decoration)
- **Dấu hiệu:** Mọi gạch đầu dòng đều có format: `**Chữ bôi đậm:** Nội dung diễn giải lặp lại`.
- **Cách xử lý:** Viết thành câu văn liền mạch hoặc chỉ nhấn mạnh từ khóa thực sự mang tính phân loại (ví dụ: tên Track, ngày giờ).

### F. Tái diễn giải ngữ cảnh người đọc đã biết (Re-explaining Known Context - Pattern #26)
- **Dấu hiệu trong thảo luận/phản hồi:** Nhắc lại từ đầu toàn bộ bối cảnh, liệt kê lại chi tiết chẩn đoán dông dài trước khi đưa ra quyết định; nhét câu trả lời chính ở tận dòng cuối.
- **Cách xử lý:** Đi thẳng vào quyết định/hướng giải quyết ngay câu đầu tiên. Chỉ giữ lại 1 sự kiện mấu chốt hoặc link cần thao tác, lược bỏ toàn bộ phần mào đầu mà đối phương đã nắm rõ.

## 2. Quy trình Humanize 4 bước

1. **Scan Tells:** Đọc qua bản thảo, đánh dấu toàn bộ cấu trúc AI (vế "không chỉ... mà còn...", câu hỏi tu từ sến, từ ngữ sáo rỗng, icon spam).
2. **Lock Facts:** Khóa toàn bộ dữ kiện gốc (ngày giờ, địa điểm, tên track, mentor, link form, nhiệm vụ). Không được lược bỏ fact và không được tự bịa fact.
3. **Rewrite with Builder Voice:** Viết lại bằng nhịp điệu của một developer/marketer thực thụ:
   - Dùng thuật ngữ công nghệ chính xác (*localhost, production, microservices, pipeline, debug, pull request*).
   - Đan xen câu ngắn và câu dài để tạo nhịp tự nhiên.
4. **Aloud Check:** Đọc to bài viết. Nếu nghe giống một sinh viên thực thụ đang nói chuyện hoặc một lead dev đang chia sẻ, bài viết đạt chuẩn.
