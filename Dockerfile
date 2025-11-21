# Stage 1: Build React UI
FROM node:20-slim AS ui-build
WORKDIR /app/ui/poker_component
COPY ui/poker_component/package*.json ./
RUN npm install
COPY ui/poker_component/ ./
RUN npm run build

# Stage 2: Python App
FROM python:3.13-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy built UI assets from Stage 1
COPY --from=ui-build /app/ui/poker_component/dist /app/ui/poker_component/dist

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0"]
