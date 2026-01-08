import { defineConfig } from 'vite'
// import react from '@vitejs/plugin-react'
import react from '@vitejs/plugin-react-swc'
// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(
      // {
      //   babel: {
      //     plugins: [['babel-plugin-react-compiler']],
      //   },
      // }
    ),
  ],
  server: {
    port: 8001,
    strictPort: true,
    host: true,
  }
})