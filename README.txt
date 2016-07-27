RRToolbox (Retinal Restauration Toolbox)

To install
 $ pyinstaller -p ./ -n imrestore -F ./imrestore.py/ --version-file=version
 
To test imrestore script:
$ python imrestore.py tests/im1*

To test imrestore executable
./dist/imrestore tests/im1*

To test imrestore executable if in dist folder:
$ ./imrestore ../tests/im1*

Read the RRtoolbox manual:
RRtoolbox.pdf

Read the disertation which made this project possible:
draft.pdf

Imrestore is an application to restore images in general but in this case is configured to restore retinal images. Because it is still in development it is the alpha program for testing and to let the users find out about its utilities. Do not hesitate to share it to the world, let everyone know how awesome it is!! (be warned that it is for research purposes)

> Thank you for downloading and be patient that not everything is given in a gold tray ;)
> --David Toro

# Aplication
A basic use would be typing './imrestore tests/im1* --lens' in the terminal which species to imrestore to load from the test path images that start with im1 specified by the wildcard '*' and the option --lens adds as its name implies lens to the retinal area.

* So, it loads this image but it presents some flares:
![Rethina photo 1]
(https://github.com/davtoh/RRtools/blob/master/tests/im1_1.jpg)

* The second image is from a different perspective of the same retina but has information that the other does not have:
![Rethina photo 2]
(https://github.com/davtoh/RRtools/blob/master/tests/im1_2.jpg)

* And Voilà they are merged into one! notice how the flares tend to disappear and the lens were added too!
![Rethina photo result]
(https://github.com/davtoh/RRtools/blob/master/tests/_restored_im1_1.jpg)

For help just type in './imrestore --help', it could not be more easier than that!

# Usage

    imrestore.py [-h] [-v VERBOSITY] [-f FEATURE] [-u POOL] [-c CACHEPATH]
                        [-e CLEARCACHE] [--loader [LOADER]] [-p [PROCESS_SHAPE]]
                        [-l [LOAD_SHAPE]] [-b [BASEIMAGE]] [-m SELECTMETHOD]
                        [-d DISTANCETHRESH] [-i INLINETHRESH]
                        [-r RECTANGULARITYTHRESH] [-j RANSACREPROJTHRESHOLD] [-n]
                        [-t] [-s [SAVE]] [-o] [-g] [-y DENOISE] [-a] [-k]
                        [-z {RetinalRestore,ImRestore}] [-x EXPERT] [-q] [-w]
                        [--onlykeys]
                        [filenames [filenames ...]]

    Restore images by merging and stitching techniques.

    positional arguments:
      filenames             List of images or path to images. Glob operations can
                            be achieved using the wildcard sign "*". It can load
                            image from files, urls, servers, stringsor directly
                            from numpy arrays (supports databases)Because the
                            shell process wildcards before it gets to the parser
                            it creates a list of filtered files in the path. Use
                            quotes in shell to prevent this behaviour an let the
                            restorer do it instead e.g. "/path/to/images/*.jpg".
                            if "*" is used then folders and filenames that start
                            with an underscore "_" are ignored by the restorer

    optional arguments:
      -h, --help            show this help message and exit
      -v VERBOSITY, --verbosity VERBOSITY
                            (0) flag to print messages and debug data. 0 -> do not
                            print messages. 1 -> print normal messages. 2 -> print
                            normal and debug messages. 3 -> print all messages and
                            show main results. (consumes significantly more
                            memory). 4 -> print all messages and show all results.
                            (consumes significantly more memory). 5 -> print all
                            messages, show all results and additional data.
                            (consumes significantly more memory).
      -f FEATURE, --feature FEATURE
                            Configure detector and matcher
      -u POOL, --pool POOL  Use pool Ex: 4 to use 4 CPUs
      -c CACHEPATH, --cachePath CACHEPATH
                            saves memoization to specified path. This is useful to
                            save some computations and use them in next
                            executions. Cached data is not guaranteed to work
                            between different configurations and this can lead to
                            unexpected program behaviour. If a different
                            configuration will be used it is recommended to clear
                            the cache to recompute values. If True it creates the
                            cache in current path.
      -e CLEARCACHE, --clearCache CLEARCACHE
                            clear cache flag.* 0 do not clear.* 1 re-compute data
                            but other cache data is left intact.* 2 All CachePath
                            is cleared before use.Notes: using cache can result in
                            unexpected behaviour if some configurations does not
                            match to the cached data.
      --loader [LOADER]     Custom loader function used to load images. By default
                            or if --loader flag is empty it loads the original
                            images in color. The format is "--loader colorflag, x,
                            y" where colorflag is -1,0,1 for BGRA, gray and BGR
                            images and the load shape are represented by x and y.
                            Ex 1: "0,100,100" loads gray images of shape (100,100)
                            in gray scale. Ex 2: "1" loads images in BGR color and
                            with original shapes. Ex 3: "0,200,None" loads gray
                            images of shape (200,None) where None is calculated to
                            keep image ratio.
      -p [PROCESS_SHAPE], --process_shape [PROCESS_SHAPE]
                            Process shape used to convert to pseudo images to
                            process features and then convert to the original
                            images. The smaller the image more memory and speed
                            gain. By default process_shape is 400,400If the -p
                            flag is empty it loads the original images to process
                            the features but it can incur to performance penalties
                            if images are too big and RAM memory is scarce
      -l [LOAD_SHAPE], --load_shape [LOAD_SHAPE]
                            shape used to load images which are beeing merged.
      -b [BASEIMAGE], --baseImage [BASEIMAGE]
                            Specify images name to use from path as first image to
                            merge in the empty restored image. By default it
                            selects the image with most features. If the -b flag
                            is empty it selects the first image in filenames as
                            base image
      -m SELECTMETHOD, --selectMethod SELECTMETHOD
                            Method to sort images when matching. This way the
                            merging order can be controlled.* (None) Best matches*
                            Histogram Comparison: Correlation, Chi-
                            squared,Intersection, Hellinger or any method found in
                            hist_map* Entropy* custom function of the form:
                            rating,fn <-- selectMethod(fns)
      -d DISTANCETHRESH, --distanceThresh DISTANCETHRESH
                            Filter matches by distance ratio
      -i INLINETHRESH, --inlineThresh INLINETHRESH
                            Filter homography by inlineratio
      -r RECTANGULARITYTHRESH, --rectangularityThresh RECTANGULARITYTHRESH
                            Filter homography by rectangularity
      -j RANSACREPROJTHRESHOLD, --ransacReprojThreshold RANSACREPROJTHRESHOLD
                            Maximum allowed reprojection error to treat a point
                            pair as an inlier
      -n, --centric         Tries to attach as many images as possible to each
                            matching. It is quicker since it does not have to
                            process too many match computations
      -t, --hist_match      Apply histogram matching to foreground image with
                            merge image as template
      -s [SAVE], --save [SAVE]
                            Customize image name used to save the restored
                            image.By default it saves in path with name
                            "_restored_{base_image}".if the -s flag is specified
                            empty it does not save. Formatting is supported so for
                            example the default name can be achived as "-s
                            {path}_restored_{name}{ext}"
      -o, --overwrite       If True and the destine filename for saving
                            alreadyexists then it is replaced, else a new filename
                            is generatedwith an index
                            "{filename}_{index}.{extension}"
      -g, --grow_scene      Flag to allow image to grow the scene so that that the
                            final image can be larger than the base image
      -y DENOISE, --denoise DENOISE
                            Flag to process noisy images. Use mild, normal, heavy
                            or provide parameters for a bilateral filter as '--
                            denoise d,sigmaColor,sigmaSpace' as for example '--
                            denoise 27,75,75'. By default it is None which can be
                            activated according to the restorer, if an empty flag
                            is provided as '--denoise' it deactivates de-noising
                            images.
      -a, --lens            Flag to apply lens to retinal area. Else do not apply
                            lens
      -k, --enclose         Flag to enclose and return only retinal area. Else
                            leaves image "as is"
      -z {RetinalRestore,ImRestore}, --restorer {RetinalRestore,ImRestore}
                            imrestore is for images in general but it can be
                            parametrized. By default it has the profile
                            "retinalRestore" for retinal images but its general
                            behaviour can be restorerd by changing it to
                            "imrestore"
      -x EXPERT, --expert EXPERT
                            path to the expert variables
      -q, --console         Enter interactive mode to let user execute commands in
                            console
      -w, --debug           Enter debug mode to let programmers find bugs
      --onlykeys            Only compute keypoints. This is useful when
                            --cachePath is used and the user wants to have the
                            keypoints cached beforehand

# Optimization techniques imrestore (oriented to retinal images):

    Restore images by merging and stitching techniques.

    resize to smaller versions*

    memoization*:
        -persistence
        -serialization and de-serialization
        -caching

    multitasking*:
        -multiprocessing
        -multithreading

    lazy evaluations:
        -load on demand
        -use of weak references

    Memory mapped files*

# STEPS:

    (1) Local features: Key-points and descriptors:
        -(1.1) SIFT, SURF, ORB, etc
        -ASIFT*

    (2) Select main or base image from set for merging:
        -Raw, Sorting, User input

    (3) Matching (spacial):
        -filter 0.7 below Hamming distance
        -key points classification

    (4) selection in matching set: (pre selection of good matches)
        (4.1) Best matches: for general purpose
        (4.2) Entropy: used when set is ensured to be of the same object
            (The program ensures that, if it is not the case).
        (4.3) Histogram comparison: use if set contains unwanted
            perspectives or images that do not correspond to image.
        (4.4) Custom function

    (5) Calculate Homography

    (6) Probability tests: (ensures that the matches images
    correspond to each other)

    (7) Merging
        (7.1) Histogram matching* (color)
        (7.2) Segmentation*
        (7.3) Alpha mask calculation*
        (7.4) Stitching and Merging

    (8) Overall filtering*:
        Bilateral filtering

    (9) Lens simulation for retinal photos*

    * optional

# Notes:

    Optimization techniques:

        Resize to smaller versions: process smaller versions of the
        inputs and convert the result back to the original versions.
        This reduces processing times, standardize the way data is
        processed (with fixed sizes), lets limited memory to be used,
        allows to apply in big images without breaking down algorithms
        that cannot do that.

        Memoization:
            Persistence: save data to disk for later use.

            serialization and de-serialization: (serialization, in
            python is refereed as pickling) convert live objects into
            a format that can be recorded; (de-serialization, in python
            referred as unpickling) it is used to restore serialized
            data into "live" objects again as if the object was created
            in program conserving its data from precious sessions.

            Caching: saves the result of a function depending or not in
            the inputs to compute data once and keep retrieving it from
            the cached values if asked.

        Multitasking:
            Multiprocessing: pass tasks to several processes using the
                computer's cores to achieve concurrency.
            Multithreading: pass tasks to threads to use "clock-slicing"
                of a processor to achieve "concurrency".

        Lazy  evaluations:
            load on demand: if data is from an external local file, it is
            loaded only when it is needed to be computed otherwise it is
            deleted from memory or cached in cases where it is extensively
            used. For remotes images (e.g. a server, URL) or in a inadequate
            format, it is downloaded and converted to a numpy format in a
            temporal local place.

            Use of weak references: in cases where the data is cached or
            has not been garbage collected, data is retrieved through
            weak references and if it is needed but has been garbage
            collected it is load again and assigned to the weak reference.

        Memory mapped files:
        Instantiate an object and keep it not in memory but in a file and
        access it directly there. Used when memory is limited or data is
        too big to fit in memory. Slow downs are negligible for read only
        mmaped files (i.e. "r") considering the gain in free memory, but
        it is a real drawback for write operations (i.e. "w","r+","w+").

    Selection algorithms:
        Histogram comparison - used to quickly identify the images that
            most resemble a target
        Entropy - used to select the best enfoqued images of the same
            perspective of an object

    Local features: Key-points and descriptors:
        ASIFT: used to add a layer of robustness onto other local
        feature methods to cover all affine transformations. ASIFT
        was conceived to complete the invariance to transformations
        offered by SIFT which simulates zoom invariance using gaussian
        blurring and normalizes rotation and translation. This by
        simulating a set of views from the initial image, varying the
        two camera axis orientations: latitude and longitude angles,
        hence its acronym Affine-SIFT. Whereas SIFT stands for Scale
        Invariant Feature Transform.

    Matching (spacial):
        Calculate Homography: Used to find the transformation matrix
        to overlay a foreground onto a background image.

    Filtering:
        Bilateral filtering: used to filter noise and make the image
        colors more uniform (in some cases more cartoonist-like)

    Histogram matching (color): used to approximate the colors from the
        foreground to the background image.

    Segmentation: detect and individualize the target objects (e.g. optic
        disk, flares) to further process them or prevent them to be altered.

    Alfa mask calculation: It uses Alfa transparency obtained with sigmoid
        filters and binary masks from the segmentation to specify where an
        algorithm should have more effect or no effect at all
        (i.e. intensity driven).

    Stitching and Merging:
        This is an application point, where all previous algorithms are
        combined to stitch images so as to construct an scenery from the
        parts and merge them if overlapped or even take advantage of these
        to restore images by completing lacking information or enhancing
        poorly illuminated parts in the image. A drawback of this is that
        if not well processed and precise information is given or calculated
        the result could be if not equal worse than the initial images.

    Lens simulation for retinal photos: As its name implies, it is a
        post-processing method applied for better appeal of the image
        depending on the tastes of the user.

>Contributions and bug reports are appreciated.
>author: David Toro
>e-mail: davsamirtor@gmail.com
>project: https://github.com/davtoh/RRtools