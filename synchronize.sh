#!/bin/bash
# Type from bash/git bash on windows ` ./synchronize.sh "commit messsage" `
# Preliminary Checks:

# Check if an argument was provided
if [ -z "$1" ]; then
    commit_message="updated content"
else
    commit_message="$1"
fi

echo "Committing with message: $commit_message"


# Git changes
git add -A
git commit -m "$commit_message"
git push
echo "Git changes pushed"

