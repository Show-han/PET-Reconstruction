import SimpleITK as sitk
import numpy as np
import os
from glob import glob

def cut_image(itk_image):
    image = np.array(sitk.GetArrayFromImage(itk_image)).transpose(2, 1, 0)
    image = image.astype(np.float32)
    image = image[156+10:284+10, 156+35:284+35, -1-128:-1]
    return image

#将2d切片合并成3d
def merge_ima(root):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(root)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image


def process_images(root, dose_label, savepath):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(root)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()

    print(root)
    image_new = sitk.GetImageFromArray(cut_image(image).transpose(2, 1, 0))
    image_new.SetOrigin(origin)
    image_new.SetDirection(direction)
    image_new.SetSpacing(spacing)
    sitk.WriteImage(image_new, savepath + "\\" + dose_label + "\\" + root.split('\\')[-2] + ".img")


savepath = 'F:\\data_ima\\ima'
imgpath = 'F:\\data_ima\\Subject_184-189\\070722_2_20220707_163946'

if __name__ == '__main__':
    labelPath = glob(imgpath)
    dose_map = {'Full_dose': 'NORMAL_1', '1-2 dose': '2', '1-4 dose': '4',
                '1-10 dose': '10', '1-20 dose': '20', '1-50 dose': '50', '1-100 dose': '100'}

    for root, _, _ in os.walk(imgpath):
        for end, dose_label in dose_map.items():
            if root.endswith(end):
                process_images(root, dose_label, savepath)
