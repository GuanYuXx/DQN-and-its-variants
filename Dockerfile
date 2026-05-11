FROM python:3.10-slim

# 讓 Python log 即時輸出到 HF 的 build log（不會卡住）
# 同時不產生 .pyc 快取檔
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 先只複製 requirements，利用 Docker layer cache 加速後續 rebuild
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再複製所有專案檔案
COPY . .

# Hugging Face Spaces 要求以 UID 1000 的非 root 使用者執行
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

EXPOSE 7860

CMD ["python", "app.py"]
