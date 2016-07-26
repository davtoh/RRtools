RRToolbox (Retinal Restauration Toolbox)

To install
 $ pyinstaller -p ./ -n imrestore -F ./imrestore.py/ --version-file=version
 
To test imrestore script:
$ python imrestore.py tests/im1*

To test imrestore executable
./dist/imrestore tests/im1*

To test imrestore executable if in dist folder:
$ ./imrestore ../tests/im1*