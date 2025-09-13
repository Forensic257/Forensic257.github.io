README - Image Alignment Notebook Instructions
==============================================

Before running the Jupyter Notebook, make sure all photos are uploaded directly into the base directory.
Do NOT place them in a separate folder. I uploaded them one at a time, and the code expects them to be in the same location.

Once the photos are in place, you can begin running the cells in the notebook.

1. SINGLE-SCALE ALIGNMENT
   - This cell processes the 3 JPG photos.
   - It outputs red and green offsets for each image.
   - Processing times are displayed.
   - Aligned images are saved into a newly created output folder.

2. MULTI-SCALE ALIGNMENT
   - This cell processes all TIFF files in the base directory.
   - It displays offsets and processing times for each TIFF file.

3. DISPLAY ALIGNED TIFF FILES
   - This cell shows the processed TIFFs.
   - You may notice border artifacts and even TWO EMIRS.
   - Most TIFFs are aligned correctly despite the quirks.

4. FIX EMIR + ADD CONTRAST
   - This cell attempts to fix Emir using the structural similarity algorithm.
   - It also enhances image contrast.
   - Outputs include:
     - Red and green offsets
     - Processing times
     - Cropped images
     - Contrast-enhanced images
   - Results are saved in two folders

5. MANUAL HARD CROP
   - This cell manually crops images from both folders.
   - Cropping is done by 10%.
   - My crop algorithm didn’t work, so I hard-cropped them manually (not in 280A).

6. FINAL DISPLAY
   - This cell shows the original, cropped-only, and contrast-enhanced images side-by-side.

End of README


