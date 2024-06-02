# Image Processing
We use skimage, scipy, pyvips, nibabel, pydicom, and highdicom as our primary image processing libraries. You're welcome to use other libraries in conjunction with the tools found here but these libraries were chosen for their balance of performance, portability, and capabilities.
## Imsuite
Imsuite contains a majority of our image processing utilities, most of these are simple wrappers or helpers for the libraries mentioned above. If you're already proficient in python, feel free to simply use the old functions you know and love. These functions are created to emulate prexisting matlab functions.
imread will read just about any image, but it will complain if there is a more specific function. For example if you read a nifti using imread it will read it, but it will complain to you saying: "Nifti detected, use niftiread() for clarity if possible otherwise enable wild. Using niftiread..." As the warning says, you should use niftiread if you know you are going to read a nifti. Here are our image IO functions:
* imread
    - Expected to read standard images into a 2d (likely multichannel) numpy array, like jpeg, tiff, or pngs. However as mentioned above can be used to read any image this library supports.
* niftiread
    - Read a .nii or .nii.gz file into a 3d single channel numpy array.
* dicomread
    - Reads a single dcm file into a 2d single channel numpy array.
* dicomread_volume
    - Reads a directory of dicoms into a 3d single channel numpy.
* wsiread
    - Reads a WSI image format like .ome.tiff or .svs into a pyvips image. Pyvips allows us to operate on a large image without immediately blowing up our memory. (but be careful!)
* imwrite
    - Writes a generic image file like .tiff or .jpeg based on the file extension. Takes keyword arguments to modify writes, but the specific keyword arguments are dependent on the file format written. Check out the pyvips docs for more info on file writing.
* 