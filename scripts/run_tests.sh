#!/bin/bash

run_tests_options="--all --backend --frontend --headed --benchmark"

_run_tests_script_completions() {
  local cur="${COMP_WORDS[COMP_CWORD]}"
  COMPREPLY=( $(compgen -W "${run_tests_options}" -- ${cur}) )
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  # Script execution logic (unchanged)
  if [ -z "$1" ]; then
    echo "Usage: $0 <${run_tests_options}>"
    exit 1
  fi

  case "$1" in
    --all)
      pytest -n auto --doctest-modules
      ;;
    --backend)
      pytest -n auto rgi --doctest-modules --benchmark-skip
      ;;
    --benchmark)
      pytest rgi/benchmarks --benchmark-only
      ;;
    --frontend)
      pytest -n auto web_app --doctest-modules
      ;;
    --headed)
      pytest web_app/tests/test_connect4_frontend.py -v --headed --doctest-modules
      ;;
    *)
      echo "Invalid option. Use: ${run_tests_options}"
      exit 1
      ;;
  esac
else
  # If sourced, define the completion function with a unique name
  complete -F _run_tests_script_completions run_tests.sh
fi
