FROM python:3.12-slim

# Системные зависимости для Chromium (нужен mmdc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg \
    chromium fonts-liberation \
    libasound2 libatk-bridge2.0-0 libatk1.0-0 libcups2 \
    libdbus-1-3 libdrm2 libgbm1 libgtk-3-0 libnspr4 libnss3 \
    libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxcursor1 \
    libxdamage1 libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 \
    libxss1 libxtst6 xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# mermaid-cli (использует системный Chromium)
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium
RUN npm install -g @mermaid-js/mermaid-cli

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py puppeteer-config.json ./
ENV PUPPETEER_CONFIG_PATH=/app/puppeteer-config.json

CMD ["python", "-u", "bot.py"]
