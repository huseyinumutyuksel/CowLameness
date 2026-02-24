import json

try:
    with open('Cow_Lameness_Analysis_v32.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    source_code = ""
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_code += "".join(cell['source'])
            
    # Checks
    nose_check = '"nose": ["nose"]' in source_code or "'nose': ['nose']" in source_code
    verbose_check = 'verbose=False' not in source_code
    csv_fix_check = 'part = str(c[1])' in source_code
    
    print(f"Check 1: Nose mapping correct? {nose_check}")
    print(f"Check 2: Verbose removed? {verbose_check}")
    print(f"Check 3: CSV flattening logic fixed? {csv_fix_check}")
    
    if nose_check and verbose_check and csv_fix_check:
        print("✅ SUCCESS: Notebook verified!")
    else:
        print("❌ FAILURE: Notebook missing fixes!")

except Exception as e:
    print(f"Error: {e}")
