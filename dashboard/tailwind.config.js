/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        cosmos: {
          950: "#040711",
          900: "#0A1022",
          800: "#111A35",
          700: "#18264A",
          600: "#25457C",
        },
        aurora: {
          cyan: "#5EF4FF",
          lime: "#B7FF68",
          amber: "#FFB84D",
          coral: "#FF6B7A",
          blue: "#6EA8FF",
        },
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(94, 244, 255, 0.18), 0 14px 36px rgba(12, 26, 58, 0.55)",
      },
      backgroundImage: {
        nebula:
          "radial-gradient(circle at 20% 20%, rgba(94, 244, 255, 0.14), transparent 45%), radial-gradient(circle at 80% 10%, rgba(183, 255, 104, 0.10), transparent 35%), radial-gradient(circle at 60% 80%, rgba(255, 184, 77, 0.08), transparent 40%), linear-gradient(155deg, #040711 10%, #0A1022 45%, #111A35 100%)",
      },
      fontFamily: {
        display: ["Orbitron", "sans-serif"],
        body: ["Manrope", "sans-serif"],
      },
    },
  },
  plugins: [],
};
