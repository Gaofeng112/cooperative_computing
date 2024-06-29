import os
import re

def update_function_names(file_path, prefix, x):
    # Read the contents of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Define the patterns to search and replace
    patterns = [
        (r'void \*csinn_\(', f'void *csinn_{prefix}_{x}('),
        (r'void csinn_update_input_and_run\(', f'void csinn_update_input_and_run_{prefix}_{x}(')
    ]

    # Perform the replacements
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    # Write the modified content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def main():
    current_dir = os.getcwd()

    # Regular expression to match the required directories
    dir_pattern = re.compile(r'^(cpu|npu)_Subgraphs_(\d+)_out$')

    for dir_name in os.listdir(current_dir):
        match = dir_pattern.match(dir_name)
        if match:
            prefix = match.group(1)
            x = match.group(2)
            model_c_path = os.path.join(current_dir, dir_name, 'model.c')

            if os.path.isfile(model_c_path):
                print(f'Updating file: {model_c_path}')
                update_function_names(model_c_path, prefix, x)
            else:
                print(f'File not found: {model_c_path}')

if __name__ == '__main__':
    main()
