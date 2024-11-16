#!/bin/bash

output_file="outputs/optimizer_implementations_in_one_file.txt"

# Clear or create the output file
> "$output_file"

# Get all .py files except sgd_curvature_memory_efficient.py and __pycache__
find optimizers -maxdepth 1 -type f -name "*.py" ! -name "sgd_curvature_memory_efficient.py" ! -path "*/\.*" | sort | while read -r file; do
    # Get just the filename without path
    filename=$(basename "$file")
    
    # Skip __init__.py and other special files
    if [[ "$filename" == "__"* ]]; then
        continue
    fi
    
    # Extract base name without .py
    base_name="${filename%.py}"
    
    # If this is not a curvature file
    if [[ ! "$base_name" == *"_curvature"* ]]; then
        # Write the regular optimizer
        echo "${filename}:" >> "$output_file"
        echo '```' >> "$output_file"
        cat "$file" >> "$output_file"
        echo '```' >> "$output_file"
        echo >> "$output_file"
        
        # Check if curvature version exists
        curvature_file="optimizers/${base_name}_curvature.py"
        if [ -f "$curvature_file" ]; then
            echo "${base_name}_curvature.py:" >> "$output_file"
            echo '```' >> "$output_file"
            cat "$curvature_file" >> "$output_file"
            echo '```' >> "$output_file"
            echo >> "$output_file"
        fi
    fi
done

echo "Generated $output_file with optimizer implementations"