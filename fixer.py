import os

# Path to your 'cubes' folder
path = r'C:\Dsg\acceleronSoln\cubes'

for filename in os.listdir(path):
    if filename.endswith(".yml"):
        file_path = os.path.join(path, filename)
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Check if primary_key already exists
        if any('primary_key: true' in line for line in lines):
            continue
            
        new_lines = []
        found_dimensions = False
        
        for line in lines:
            new_lines.append(line)
            if 'dimensions:' in line and not found_dimensions:
                # Get the indent of the 'dimensions' line
                indent = len(line) - len(line.lstrip()) + 2
                # Add the primary key block
                new_lines.append(' ' * indent + "- name: id\n")
                new_lines.append(' ' * indent + "  sql: id\n")
                new_lines.append(' ' * indent + "  type: number\n")
                new_lines.append(' ' * indent + "  primary_key: true\n")
                found_dimensions = True
        
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
            print(f"✅ Patched {filename}")
