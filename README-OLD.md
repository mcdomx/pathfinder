# pathfinder
Deep Learning application used to find a path in natural landscape settings

This application uses Keras deep learning library to train a model to identify navigable areas in an off-road setting.  

The model is trainde on dat from http://www.mikeprocopio.com/labeledlagrdata.html.

Downloaded data includes 6 training videos and 3 testing videos.  Each downloaded video is represented by a downloaded directory that includes a series of *.mat files eah of which represents a single frame from a video.

The training directories need to be moved to a user-created directory that only includes the downloaded training directories.  The test directories need to be moved to a different user-created directory that includes only the 3 downloaded testing directories.

The pathfinder_master_notebook will ask for the local path to these two directries as well as a third directory that will be used by the application to store images, masks and video files needed to train and review the results. 
