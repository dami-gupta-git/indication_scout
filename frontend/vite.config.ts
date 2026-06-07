import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: proxy /api to the uvicorn backend so the frontend and API share an origin.
// In Docker the API is reached at the "api" service host (VITE_API_PROXY_TARGET);
// on the host it defaults to localhost:8000.
const apiTarget = process.env.VITE_API_PROXY_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
});
