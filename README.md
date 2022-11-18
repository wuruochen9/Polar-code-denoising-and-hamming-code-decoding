# bert-based polar code noise reduction and hamming code decoding.  

bert is widely used in NLP problems. But we can consider hamming code decoding as a process of translation. Considering this, I try to use transformer solving code     decoding problem, which including polar code channel decoding and hamming code decoding.
### Here's the whole process of my code.
First, we need to create original 8 bits codes like 00000000. After polar code channel encoding and noise added, the codes is like 0.2 0.4 0.3...0.1(16bits). 
Then we apply bert to restore it to the code after encodeing.  
## bert based hamming code decoding.ipynb is for hamming code decoding
using the same process, I decode the 12 bits hamming code to 8 bits original codes. But this time there's no noise
## Result
The result is not very good. The accuracy of hamming_code is about 96% and accuracy of polar_code is about 99.996% when SNR=10.0. But I'll try to update my method to improve the result.
