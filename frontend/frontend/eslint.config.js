import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";

export default [
  {
    ignores: ["dist", "node_modules"]
  },

  js.configs.recommended,

  {
    files: ["**/*.{js,jsx}"],

    languageOptions: {
      ecmaVersion: 2022,

      globals: {
        ...globals.browser
      },

      parserOptions: {
        ecmaFeatures: {
          jsx: true
        }
      }
    },

    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh
    },

    rules: {
      "no-unused-vars": [
        "error",
        {
          "varsIgnorePattern": "^[A-Z_]"
        }
      ],

      "react-hooks/rules-of-hooks": "error",

      "react-hooks/exhaustive-deps": "warn",

      "react-refresh/only-export-components": "warn"
    }
  }
];