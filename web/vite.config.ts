import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],

  // Relative, so the built bundle works when FastAPI serves it from any path.
  base: "./",

  server: {
    // host:true binds to the LAN, not just loopback -- without it the phone
    // cannot load this page at all.
    host: true,
    proxy: {
      // The phone loads this page from the laptop's LAN address, so a direct
      // call to localhost:8000 would resolve to the PHONE. Proxying keeps the
      // API same-origin: no CORS middleware, no hardcoded IP to update every
      // time DHCP moves the laptop.
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
