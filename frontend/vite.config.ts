import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: proxy /api to the uvicorn backend so the frontend and API share an origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
