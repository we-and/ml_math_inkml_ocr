#!/bin/bash

# Check if the correct number of arguments was provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

input_path="$1"
output_path="$2"

# Ensure the output directory exists
mkdir -p "$output_path"

# Loop through all .inkml files in the input directory
for input_file in "$input_path"/*.inkml; do
    # Extract the base filename without the extension
    base_name=$(basename "$input_file" .inkml)

    # Define the output file path
    output_file="$output_path/${base_name}.png"

    # Placeholder for the actual processing command
    # Here you would call your processing application or script
    echo "Processing $input_file to $output_file"
#python3 ../inkml2img-master/inkml2img.py test_inputs/129_em_542.inkml test_inputs_png/129_em_542.pn
	python3 ../inkml2img-master/inkml2img.py test_inputs/$base_name.inkml $output_file
    # Example: python my_converter.py "$input_file" "$output_file"
done
