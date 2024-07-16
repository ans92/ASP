python3 "/home/ans/CZSL/KG-SP-main/train.py" --config "/home/ans/CZSL/KG-SP-main/configs/kgsp/utzappos.yml" --open_world --fast

python3 "/home/ans/CZSL/KG-SP-main/train.py" --config "/home/ans/CZSL/KG-SP-main/configs/kgsp/mit.yml" --open_world --fast

python3 "/home/ans/CZSL/KG-SP-main/train.py" --config "/home/ans/CZSL/KG-SP-main/configs/kgsp/cgqa.yml" --open_world --fast

python3 "/home/ans/CZSL/KG-SP-main/test.py" --logpath "/home/ans/CZSL/logs/kgsp/mitstates/" --open_world --fast

python3 "/home/ans/CZSL/KG-SP-main/test.py" --logpath "/home/ans/CZSL/logs/kgsp/cgqa/" --open_world --fast

python3 "/home/ans/CZSL/KG-SP-main/test.py" --logpath "/home/ans/CZSL/logs/kgsp/utzappos/" --open_world --fast