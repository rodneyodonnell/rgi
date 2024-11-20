#!/bin/bash

run_linter_options="--mypy --pylint --eslint --tsc --python --all"

_run_linter_script_completions() {
  local cur="${COMP_WORDS[COMP_CWORD]}"
  COMPREPLY=( $(compgen -W "${run_linter_options}" -- ${cur}) )
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  # Script execution logic (unchanged)
  if [ -z "$1" ]; then
    echo "Usage: $0 <${run_linter_options}>"
    exit 1
  fi

  declare -A commands
  commands["--mypy"]="mypy ."
  commands["--pylint"]="pylint --rcfile=pyproject.toml \$(git ls-files '*.py')"
  commands["--eslint"]="yarn eslint 'web_app/src/**/*.ts'"
  commands["--tsc"]="yarn tsc --noEmit"

  selected_commands=()
  for arg in "$@"; do
    case "$arg" in
      --mypy|--pylint|--eslint|--tsc)
        selected_commands+=("${commands[$arg]}")
        ;;
      --python)
        selected_commands+=("${commands[--mypy]}" "${commands[--pylint]}")
        ;;
      --all)
        for cmd in "${!commands[@]}"; do
          selected_commands+=("${commands[$cmd]}")
        done
        ;;
      *)
        echo "Invalid option. Use: ${run_linter_options}"
        exit 1
        ;;
    esac
  done

  for cmd in "${selected_commands[@]}"; do
    eval "$cmd"
  done
else
  # If sourced, define the completion function with a unique name
  complete -F _run_linter_script_completions run_linter.sh
fi