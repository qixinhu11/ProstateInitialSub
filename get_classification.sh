python dicom2niih5.py
cd segmentation/organ_segmentation

python test.py
cd ../..

python preprocessing.py
cd classification

python vote_for_resutls.py