import eslintPluginTypeScript from '@typescript-eslint/eslint-plugin';
import parser from '@typescript-eslint/parser';

export default [
  {
    files: ['web_app/src/**/*.ts'],
    languageOptions: {
      ecmaVersion: 2021,
      sourceType: 'module',
      parser: parser,
    },
    rules: {
      semi: ['error', 'never'],
      quotes: ['error', 'single'],
      'prefer-const': 'error',
    },
    plugins: {
      '@typescript-eslint': eslintPluginTypeScript,
    },
  },
];
