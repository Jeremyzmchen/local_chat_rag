FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production: serve with nginx
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html

# Proxy /api → backend:8000
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
