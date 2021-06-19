

def print_hi(name):
    print(f'Hi, {name}')



if __name__ == '__main__':
    print_hi('FYP - CV Based Defect Detection of Printed Embossed Fabrics')

exec(open('BR_Module.py').read())
# exec(open('Seg_Module.py').read())
# exec(open('QA_Module.py').read())

# //Folder Structure
#
# “Assets” folder---->
#
#  BR Module----Input---ref
# 					test
# Output---ref---isolated_artwork
# Test---isolated_artwork
# Mask---ref
#         ---		test
#
# Seg Module---Input---
#  Output---Matching--ref, test
# 					Non_Matching--ref,test
#
#
# QA Module---Input, Output--ref, test
#
# //Background Removal Module Inputs
#
# Upload the ref image to the BR_Module/Input/ref/ref_image.jpg
# Upload the test image to the BR_Module/Input/test/test_image.jpg
#
#
#
# //Background Removal Module Execution
#
#
# //Segmentation Module Inputs
#
# The isolated artwork of the ref image (background-removed image) - from “assets/BR_Module/Output/ref/isolated_artwork
#
# The isolated artwork of the test image -from “assets/BR_Module/Output/test/isolated_artwork”
#
# //Segmentation Module Execution
#
#
#
#
# Output=> Segment Correspondence matching
#
#
#
# //QA Module Inputs
#
# “Seg_Module/Output/Matching/ref/”
# “Seg_Module/Output/Matching/test”
#
# “Seg_Module/Output/Non_Matching/ref/”
# “Seg_Module/Output/Non_Matching/test/”
#
#
# //QA Module Execution
#
#
# //QA Module Outputs
#
# Defect Report
# (To be displayed in browser)


