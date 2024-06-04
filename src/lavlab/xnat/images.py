import getpass
import io
import os

import highdicom as hd
import xnat

from lavlab import imsuite, xnat

LOGGER = xnat.LOGGER.getChild("images")


def get_file(file_obj):
    stream = io.BytesIO()
    file_obj.download_stream(stream)
    stream.seek(0)
    return stream


def pull_file(file_obj, output_dir="."):
    path = os.path.join(output_dir, file_obj.name)
    file_obj.download(path)
    return path


def _get_scan_vol_dicom(scan_obj):
    dicom_file_resources = scan_obj.resources.get("DICOM").files
    dcms = [get_file(dicom_file_resources[dcm]) for dcm in dicom_file_resources]
    return imsuite.dicomread_volume(dcms)


def get_scan_vol(scan):
    nifti_resource = scan.resources.get("NIFTI")
    if nifti_resource is None:
        LOGGER.info("Scan does not have a NIFTI, trying DICOM")
        return _get_scan_vol_dicom(scan)
    if len(nifti_resource.files) > 1:
        LOGGER.warning("More than one NIFTI file found, using first")
    nifti_file = nifti_resource.files[0]
    nifti_stream = get_file(nifti_file)
    return imsuite.niftiread(nifti_stream)
