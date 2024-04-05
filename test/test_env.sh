#!/bin/bash

if [ ! -f demo.jpg ]; then
  wget -O demo.jpg https://github.com/open-mmlab/mmrotate/blob/main/demo/dota_demo.jpg
fi

cd /workspaces/prototype/deployment_files

required_folders="oriented_rcnn sr"
missing=false

for folder in $required_folders; do
  if [ ! -d "$folder" ]; then
    echo "Missing $folder"
    missing=true
  fi
done


if $missing; then
  exit 1
fi

if python /workspaces/prototype/models/test_oriented_rcnn.py; then
  echo "MMDeploy suceeded \n"
  first_script_success=true
else
  echo "MMDeploy failed \n"
  first_script_success=false
fi

if python /workspaces/prototype/models/test_rsinet.py; then
  echo "RSI-Net suceeded \n"
  second_script_success=true
else
  echo "RSI-Net failed \n"
  second_script_success=false
fi

if [ "$first_script_success" = true ] && [ "$second_script_success" = true ]; then
  echo "ENV setup correctly"
fi