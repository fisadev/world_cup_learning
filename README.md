world_cup_learning
==================

Just because I wanted to win a bet, some machine learning about the fifa world cup :)

Still working on it, should be ready before the 2014 cup starts playing (I will try to "predict" matches results).

And I have very little idea about football, so if you know more about the topic and have some suggestions, you are welcome to write me!
(my gmail user is the same than github)

You can see the main code and results (including some neat graphs) here: http://nbviewer.ipython.org/github/fisadev/world_cup_learning/blob/master/learn.ipynb

To run this on your machine, you will need to install the requirements in the ``requirements.txt`` file using pip (inside a virtualenv if you can, as usual, to avoid sudo), but also this packages not via pip, but apt or the tool you use in your system for compiled packages:

* numpy
* scipy

Under ubuntu you would run:

    sudo apt-get install python-numpy python-scipy
    
Then, inside a virtualenv with access to your system site-packages:

    pip install -r requirements.txt
    
(Or, if you don't want to use virtualenv, the same but with sudo)
   
