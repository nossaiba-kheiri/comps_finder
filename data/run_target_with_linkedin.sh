#!/bin/bash

# Run Target Creation with LinkedIn - Using 12 months to get Nov 2024 posts

cd "$(dirname "$0")"

echo "=================================================================================="
echo "Creating Target with LinkedIn (12 months back to include Nov 2024 posts)"
echo "=================================================================================="
echo

python create_target_from_info.py \
  --name "Huron Consulting Group Inc." \
  --url "http://www.huronconsultinggroup.com/" \
  --description "Huron Consulting Group Inc. provides consultancy and managed services in the United States and internationally. It operates through three segments: Healthcare, Education, and Commercial. The company offers financial and operational performance improvement consulting services; digital offerings; spanning technology and analytic-related services, including enterprise health record, enterprise resource planning, enterprise performance management, customer relationship management, data management, artificial intelligence and automation, technology managed services, and a portfolio of software products; organizational transformation; revenue cycle managed services and outsourcing; financial and capital advisory consulting; and strategy and innovation consulting. It also provides digital offerings; spanning technology and analytic-related services; technology managed services; research-focused consulting; managed services; and global philanthropy consulting services, as well as Huron Research product suite, a software suite designed to facilitate and enhance research administration service delivery and compliance. In addition, the company offers digital services, software products, financial capital advisory services, and Commercial consulting." \
  --primary-industry-classification "Research and Consulting Services" \
  --linkedin-url "https://www.linkedin.com/company/huronconsulting/" \
  --months-back 18

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo
    echo "=================================================================================="
    echo "Verifying LinkedIn Data"
    echo "=================================================================================="
    echo
    
    python -c "
import json
with open('target_huron_consulting_group_inc.json') as f:
    target = json.load(f)
metadata = target.get('metadata', {})
print(f'✓ LinkedIn posts fetched: {metadata.get(\"linkedin_posts_fetched\", 0)}')
print(f'✓ LinkedIn posts relevant: {metadata.get(\"linkedin_posts_relevant\", 0)}')
print(f'✓ Profile text length: {metadata.get(\"profile_text_length\", 0):,} characters')
"
fi

