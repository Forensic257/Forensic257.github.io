Assuming you have the calibration and object folders in the same directory as the code, here's a guideline of how to run my code:
- Starting with Part 0, you may start with the main.ipynb for primarily parts 0.1-0.3. For Part 0.4, VSC got a memory error so I had to switch to Google Colab to use stronger GPU computing power (Part0.py).
- Okay, by now, you should have 'my_nerf_data.npz' after a few minutes of running Part0.py. It's very large, so I had a separate file (sanity.py) to donwsize those images from 4032x3024 to 240x320. It will output you 'my_nerf_data_downsampled.npz.'
- Moving to Part 1, you may use Part1.py as VSC was able to run it fine. Though, I did use the beginning of Project_4.ipynb to run it too. Both have the same outputs.
- For Part 2, use Project_4.ipynb. VSC completely fails in terms of memory issues and strong GPU computing power is needed. I ended up having to buy the $9.99 as I go plan :(. 
