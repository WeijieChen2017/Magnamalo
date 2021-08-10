import os
import glob

nii_list = glob.glob("./*.nii")
nii_list.sort()

for nii_path in nii_list:
    cmd_gzip = "gzip " + nii_path
    cmd_rm = "rm -f " + nii_path
    for cmd in [cmd_gzip, cmd_rm]:
        print(cmd)
        os.system(cmd)