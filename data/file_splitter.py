import json

input_file = 'company_data/applicants.json'
output_file1 = 'company_data/applicants_part1.json'
output_file2 = 'company_data/applicants_part2.json'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

if isinstance(data, list):
    half = len(data) // 2
    part1 = data[:half]
    part2 = data[half:]
elif isinstance(data, dict):
    keys = list(data.keys())
    half = len(keys) // 2
    keys1 = keys[:half]
    keys2 = keys[half:]
    part1 = {k: data[k] for k in keys1}
    part2 = {k: data[k] for k in keys2}
else:
    raise ValueError("Unsupported JSON structure")

with open(output_file1, 'w', encoding='utf-8') as f1:
    json.dump(part1, f1, ensure_ascii=False, indent=2)

with open(output_file2, 'w', encoding='utf-8') as f2:
    json.dump(part2, f2, ensure_ascii=False, indent=2)

print(f"Split into {output_file1} and {output_file2}")