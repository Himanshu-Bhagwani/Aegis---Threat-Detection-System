# ─────────────────────────────────────────────
# AEGIS / Apeilo frontend  (Next.js 14)
# ─────────────────────────────────────────────
FROM node:20-alpine

WORKDIR /app

# Install deps first (better layer caching)
COPY package.json package-lock.json* ./
RUN npm config set fetch-retry-mintimeout 20000 && \
    npm config set fetch-retry-maxtimeout 120000 && \
    npm config set fetch-retries 5 && \
    if [ -f package-lock.json ]; then npm ci; else npm install; fi

# App source
COPY . .

EXPOSE 3000

# `npm run dev` keeps hot-reload + matches the workflow you mentioned
CMD ["npm", "run", "dev", "--", "--hostname", "0.0.0.0"]
