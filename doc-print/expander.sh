#!/bin/bash
# Usage: ./expand_tex.sh input.tex output.tex

input_file="$1"
output_file="$2"

if [ -z "$input_file" ] || [ -z "$output_file" ]; then
    echo "Usage: $0 <input.tex> <output.tex>"
    exit 1
fi

expand_tex() {
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "Warning: File '$file' not found!" >&2
        return
    fi

    local file_dir="$(dirname "$file")"

    while IFS= read -r line || [ -n "$line" ]; do
        # Match \input{filename} or \input{filename.tex}
        if [[ "$line" =~ \\input\{([^}]+)\} ]]; then
            local matched_file="${BASH_REMATCH[1]}"
            
            # Append .tex extension if omitted
            if [[ ! "$matched_file" =~ \.tex$ ]]; then
                matched_file="${matched_file}.tex"
            fi
            
            # Check relative to current file's directory first, then CWD
            local target_path="$file_dir/$matched_file"
            if [ ! -f "$target_path" ]; then
                target_path="$matched_file"
            fi

            if [ -f "$target_path" ]; then
                # Recursively expand nested inputs
                expand_tex "$target_path"
            else
                echo "Warning: Could not resolve input file '$matched_file'." >&2
                echo "$line"
            fi
        else
            echo "$line"
        fi
    done < "$file"
}

# Run the expansion and pipe to the output file
expand_tex "$input_file" > "$output_file"
echo "Successfully expanded '$input_file' into '$output_file'."
