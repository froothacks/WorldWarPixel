import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backend = process.env.BACKEND_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/runs": { target: backend, ws: true },
      "/rosters": backend,
      "/health": backend,
      "/models": backend,
    },
  },
});
