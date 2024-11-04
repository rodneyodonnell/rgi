import eslintPluginTypeScript from '@typescript-eslint/eslint-plugin';
import parser from '@typescript-eslint/parser';

export default [
  {
    files: ['web_app/static/**/*.ts'],
    languageOptions: {
      ecmaVersion: 2021,
      sourceType: 'module',
      parser: parser,
    },
    rules: {
      semi: ['error', 'always'],
      quotes: ['error', 'double'],
      'no-unused-vars': 'warn',
      'prefer-const': 'error',
    },
    plugins: {
      '@typescript-eslint': eslintPluginTypeScript,
    },
  },
];
